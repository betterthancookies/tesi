"""Decisione client-driven di (E, B): la Fase 5 della proposta.

Il server propone un ENVELOPE (Fase 4), il client sceglie dentro l'envelope
con informazioni che solo lui ha nel momento in cui inizia (late binding):

    SoC attuale e stato di carica   -> quanto lavoro puo' permettersi
    workload attuale                -> quanto durera' quel lavoro
    loss del modello globale sui
    propri dati                     -> quanto lavoro vale la pena di fare

Niente torch ne' flwr qui: sono funzioni pure, testabili da sole.

LA REGOLA, in quattro passi
1. Utilita' statistica  u = min(1, L / ln C)
   L e' la loss del modello globale su un campione dei dati locali, ln C la
   loss di un classificatore che tira a caso su C classi. u ~ 1 vuol dire che
   il modello globale non ha ancora imparato nulla di questi dati.
   [B] nessun parametro: ln C e' il riferimento naturale della cross-entropy.

2. Riserva disponibile  h = (SoC - soc_min) / (1 - soc_min), in [0, 1]
   h = 1 se il device e' in carica: la batteria non paga il training.

3. Epoche desiderate  E* = E_lo + round((E_hi - E_lo) * u * h)
   Il prodotto u*h e' la tabella della proposta: tanta loss E tanta carica
   -> tante epoche; poca loss O poca carica -> poche epoche.

4. Vincoli, verificati sulla TRAIETTORIA, non sullo stato attuale:
   - energia:  SoC previsto A FINE ROUND (training + upload) >= soc_min
   - deadline: durata prevista col workload ATTUALE <= deadline del server
   B "piu' economico per passo di SGD": si parte dal B piu' piccolo
   dell'envelope e si scende da E* verso E_lo finche' i vincoli sono
   rispettati; solo se a quel B non ci sta nemmeno E_lo si passa al B
   successivo (piu' grande: epoca piu' corta e meno cara).
   Se a nessun B la deadline e' rispettabile si fa E_lo col B piu' veloce e
   lo si dichiara. Se nemmeno E_lo e' sostenibile per la batteria, il client
   si ritira dal round (opt-out) senza toccare la riserva.

[B] PERCHE' IL B PIU' PICCOLO. Con la potenza di training costante per tier
l'energia per CAMPIONE cala al crescere di B (meno costo fisso per batch), ma
quella per PASSO di SGD cresce: con la stessa energia un B piccolo fa molti
piu' passi. Lo sweep su B (experiment.toml, smart_batch_min) mostra
l'accuracy crescere al calare di B; il B minimo dell'envelope (16) e' il
compromesso con il SoC finale. Il client riduce prima E e poi sale di B:
cosi' B varia davvero solo per i client lenti, occupati o scarichi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from device.fuel_gauge import DeviceReading

# chiavi del ConfigRecord con cui viaggia l'envelope
K_E_LO = "smart-e-lo"
K_E_HI = "smart-e-hi"
K_BATCHES = "smart-batches"
K_DEADLINE = "smart-deadline-s"
K_SOC_MIN = "smart-soc-min"
K_PROBE = "smart-probe-samples"
K_REFUSE_DL = "smart-refuse-deadline"
K_REFUSE_LOAD = "smart-refuse-load-only"


@dataclass(frozen=True)
class Envelope:
    """Proposta del server: dentro questi limiti decide il client.

    refuse_deadline: il client rifiuta la proposta anche se E_lo non sta
    nella deadline col workload di adesso (vedi accepts_proposal).
    refuse_load_only: con refuse_deadline, rifiuta per la deadline SOLO se
    a riposo (workload 0) ci starebbe: il ritardo e' colpa del carico del
    momento. Viaggiano nel messaggio solo se accesi.
    """

    e_lo: int
    e_hi: int
    batches: tuple[int, ...]
    deadline_s: float
    soc_min: float
    probe_samples: int
    refuse_deadline: bool = False
    refuse_load_only: bool = False

    def to_config(self) -> dict:
        cfg = {
            K_E_LO: int(self.e_lo),
            K_E_HI: int(self.e_hi),
            K_BATCHES: [int(b) for b in self.batches],
            K_DEADLINE: float(self.deadline_s),
            K_SOC_MIN: float(self.soc_min),
            K_PROBE: int(self.probe_samples),
        }
        if self.refuse_deadline:
            cfg[K_REFUSE_DL] = 1
        if self.refuse_load_only:
            cfg[K_REFUSE_LOAD] = 1
        return cfg

    @classmethod
    def from_config(cls, cfg) -> "Envelope":
        return cls(
            e_lo=int(cfg[K_E_LO]),
            e_hi=int(cfg[K_E_HI]),
            batches=tuple(int(b) for b in cfg[K_BATCHES]),
            deadline_s=float(cfg[K_DEADLINE]),
            soc_min=float(cfg[K_SOC_MIN]),
            probe_samples=int(cfg[K_PROBE]),
            refuse_deadline=bool(int(cfg[K_REFUSE_DL])) if K_REFUSE_DL in cfg else False,
            refuse_load_only=bool(int(cfg[K_REFUSE_LOAD])) if K_REFUSE_LOAD in cfg else False,
        )

    @staticmethod
    def present_in(cfg) -> bool:
        return K_E_LO in cfg


@dataclass(frozen=True)
class Plan:
    optout: bool
    epochs: int = 0
    batch: int = 0
    time_s: float = 0.0
    soc_end: float = 0.0
    deadline_ok: bool = True
    utility: float = 0.0
    e_target: int = 0
    reason: str = ""


def statistical_utility(loss: float, n_classes: int = 10) -> float:
    if not math.isfinite(loss):
        return 1.0
    return float(min(1.0, max(0.0, loss / math.log(n_classes))))


def headroom(reading: DeviceReading, soc_min: float) -> float:
    if reading.charging:
        return 1.0
    return float(min(1.0, max(0.0, (reading.soc - soc_min) / max(1e-9, 1.0 - soc_min))))


def _energy_ok(reading: DeviceReading, soc_end: float, soc_min: float) -> bool:
    return reading.charging or soc_end >= soc_min


def can_afford_minimum(reading: DeviceReading, env: Envelope, n_samples: int) -> bool:
    """Il round minimo (E_lo, batch migliore, stima della loss inclusa) lascia
    la riserva intatta? Si verifica PRIMA della stima della loss: se la
    risposta e' no, non vale la pena spendere nemmeno quella."""
    if reading.charging:
        return True
    for b in env.batches:
        _, soc_end = reading.predict(env.e_lo, b, n_samples, env.probe_samples)
        if soc_end >= env.soc_min:
            return True
    return False


def accepts_proposal(reading: DeviceReading, env: Envelope, n_samples: int) -> bool:
    """Controllo di accettazione della proposta (Fase 2b di sage_smart).

    Il client accetta se riesce a fare almeno il lavoro minimo dell'envelope
    senza intaccare la riserva, con lo stato di ADESSO. E' lo stesso
    controllo che fara' di nuovo al momento del training: qui serve a
    rifiutare PRIMA di scaricare il modello, cosi' il server puo' chiamare
    un altro client nello stesso round.

    [B] 05/11, con env.refuse_deadline: rifiuta anche se E_lo non sta nella
    deadline nemmeno col B piu' veloce, col workload di ADESSO. Senza, un
    client molto carico accetta e poi lavora fuori deadline: allunga il round
    e l'idle di tutti (esperimento sul workload, x3: round medio 455 s contro
    181 s a x1, 98% degli envelope limitati dalla deadline). Il workload e'
    l'informazione che ha solo il client; al training la regola non cambia
    (il modello e' gia' scaricato, vedi decide).

    [B] 06/11, con anche env.refuse_load_only: rifiuta per la deadline solo
    se A RIPOSO (workload 0) ci starebbe, cioe' se il ritardo e' colpa del
    carico di adesso ("sono occupato, chiedimelo dopo"). Chi non ci sta
    nemmeno a riposo (telefono lento con molti dati) accetta come nella v2:
    col solo refuse_deadline rifiutava sempre e non partecipava mai (x1: 8
    client low mai selezionati, Gini delle partecipazioni 0.32 -> 0.43).
    """
    if not can_afford_minimum(reading, env, n_samples):
        return False
    if env.refuse_deadline and not can_meet_deadline(reading, env, n_samples):
        if env.refuse_load_only and not can_meet_deadline(at_rest(reading), env, n_samples):
            return True
        return False
    return True


def at_rest(reading: DeviceReading) -> DeviceReading:
    """La stessa lettura con workload 0: il device senza altre app."""
    return replace(reading, workload=0.0)


def can_meet_deadline(reading: DeviceReading, env: Envelope, n_samples: int) -> bool:
    """E_lo col B piu' veloce dell'envelope sta nella deadline, col workload
    attuale? Stessa durata (training + stima della loss) che usa decide."""
    return min(reading.work_time_s(env.e_lo, b, n_samples, env.probe_samples)
               for b in env.batches) <= env.deadline_s


def decide(reading: DeviceReading, env: Envelope, n_samples: int,
           loss: float, n_classes: int = 10) -> Plan:
    """Sceglie (E, B) dentro l'envelope. Vedi la docstring del modulo."""
    u = statistical_utility(loss, n_classes)
    h = headroom(reading, env.soc_min)
    span = max(0, env.e_hi - env.e_lo)
    e_target = env.e_lo + int(round(span * u * h))
    p = env.probe_samples

    def options(e: int):
        out = []
        for b in env.batches:
            t, soc_end = reading.predict(e, b, n_samples, p)
            if _energy_ok(reading, soc_end, env.soc_min):
                out.append((b, t, soc_end))
        return out

    # B dal piu' piccolo: dentro ogni B si riduce E, e si sale di B solo se
    # a quel B non ci sta nemmeno E_lo
    for b in sorted(env.batches):
        for e in range(e_target, env.e_lo - 1, -1):
            t, soc_end = reading.predict(e, b, n_samples, p)
            if _energy_ok(reading, soc_end, env.soc_min) and t <= env.deadline_s:
                return Plan(False, e, b, t, soc_end, True, u, e_target,
                            "target" if e == e_target else "ridotto")

    # nessun E rispetta la deadline: E_lo col batch piu' veloce, se la
    # batteria lo regge. La deadline e' un vincolo di comparabilita', la
    # riserva no.
    feasible = options(env.e_lo)
    if feasible:
        b, t, soc_end = min(feasible, key=lambda o: o[1])
        return Plan(False, env.e_lo, b, t, soc_end, False, u, e_target,
                    "fuori-deadline")
    return Plan(True, utility=u, e_target=e_target, reason="riserva")


def affordable_steps(reading: DeviceReading, env: Envelope, n_samples: int,
                     batch: int, cap: int) -> int:
    """Massimo numero di passi di SGD che lasciano SoC finale >= soc_min.

    E' il S_max della proposta (budget energetico). Serve da guardia durante
    il training: se la previsione fosse ottimistica, il client si fermerebbe
    qui e invierebbe un update parziale invece di intaccare la riserva.
    """
    if reading.charging:
        return cap
    spe = max(1, math.ceil(n_samples / batch))

    def ok(steps: int) -> bool:
        _, soc_end = reading.predict(steps / spe, batch, n_samples,
                                     env.probe_samples)
        return soc_end >= env.soc_min

    if ok(cap):
        return cap
    lo, hi = 0, cap          # ok(lo) e' vero se il minimo era sostenibile
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo
