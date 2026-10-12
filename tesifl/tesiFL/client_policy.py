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
   Si scende da E* verso E_lo finche' un batch dell'envelope li rispetta
   entrambi; a parita' di E si sceglie il B che lascia piu' carica (in
   carica: il piu' veloce).
   Se solo la deadline non e' rispettabile si fa E_lo col B piu' veloce e lo
   si dichiara. Se nemmeno E_lo e' sostenibile per la batteria, il client si
   ritira dal round (opt-out) senza toccare la riserva.

[B] PERCHE' IL WORKLOAD SPOSTA B. Col throughput ridotto da altre app il
calcolo per campione dura di piu', il costo fisso per batch pesa meno e il
batch energeticamente ottimo si sposta. Quando invece e' la deadline a
mordere, conviene il batch piu' grande (meno passi, meno overhead): e'
l'indicazione della proposta "high workload -> larger B", che qui emerge dai
vincoli invece di essere scritta come regola.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from device.fuel_gauge import DeviceReading

# chiavi del ConfigRecord con cui viaggia l'envelope
K_E_LO = "smart-e-lo"
K_E_HI = "smart-e-hi"
K_BATCHES = "smart-batches"
K_DEADLINE = "smart-deadline-s"
K_SOC_MIN = "smart-soc-min"
K_PROBE = "smart-probe-samples"


@dataclass(frozen=True)
class Envelope:
    """Proposta del server: dentro questi limiti decide il client."""

    e_lo: int
    e_hi: int
    batches: tuple[int, ...]
    deadline_s: float
    soc_min: float
    probe_samples: int

    def to_config(self) -> dict:
        return {
            K_E_LO: int(self.e_lo),
            K_E_HI: int(self.e_hi),
            K_BATCHES: [int(b) for b in self.batches],
            K_DEADLINE: float(self.deadline_s),
            K_SOC_MIN: float(self.soc_min),
            K_PROBE: int(self.probe_samples),
        }

    @classmethod
    def from_config(cls, cfg) -> "Envelope":
        return cls(
            e_lo=int(cfg[K_E_LO]),
            e_hi=int(cfg[K_E_HI]),
            batches=tuple(int(b) for b in cfg[K_BATCHES]),
            deadline_s=float(cfg[K_DEADLINE]),
            soc_min=float(cfg[K_SOC_MIN]),
            probe_samples=int(cfg[K_PROBE]),
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
    """
    return can_afford_minimum(reading, env, n_samples)


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

    for e in range(e_target, env.e_lo - 1, -1):
        on_time = [o for o in options(e) if o[1] <= env.deadline_s]
        if on_time:
            # piu' carica residua; in carica sono tutte uguali -> il piu' veloce
            b, t, soc_end = max(on_time, key=lambda o: (o[2], -o[1]))
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
