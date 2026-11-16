from __future__ import annotations

import math
from logging import INFO

import numpy as np
import pulp
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.common import log

from device.battery_model import energy_wh_linear
from device.communication_model import comm_power_w, comm_time_s
from device.compute_model import probe_time_s, training_time_s
from device.communication_model import ctrl_latency_s
from device.constants import CHARGE_C_RATE, RECHARGE_RATE_PER_H, V_NOMINAL
from device.fuel_gauge import DeviceReading
from device.workload import stationary_utilization
from device.world_state import (
    BATTERY_NAMES, battery_delta_soc, effective_profile, idle_power_w,
    training_power_w,
)
from strategies.sage_ablation import PhysicalSAGEAblation
from tesiFL.client_policy import Envelope

# passo massimo con cui il server integra la previsione (passo 1): il mondo
# scarica l'idle una volta per round, e un round dura 100-300 s
_ESTIMATE_STEP_S = 300.0


class PhysicalSAGESmart(PhysicalSAGEAblation):
    """SAGE-smart: selezione lato server, (E, B) decisi dal CLIENT.

    [B] 16/11: questo e' l'UNICO SAGE-smart. Fino al 15/11 si chiamava
    "SAGE-smart 2" (etichetta sage_smart2_socfc, token
    partial+enrollmean+socfc+pick+nobcast); qui sono i default, senza token.
    Le versioni di prima sono state tolte dal codice: negoziazione con
    rimpiazzi senza rosa, stima del solo idle e col workload medio, rifiuto
    per deadline (dlrefuse, dlload), previsione che pesa la classifica per
    1 - w, FedNova, carico massimo all'iscrizione, broadcast ai primi M.
    Restano nella storia di git e nelle cartelle dei risultati fino alla v2.
    Fra sage_smart_lin, _peuk e _nm cambia SOLO la batteria dei device del
    mondo; l'algoritmo e' lo stesso.

    Il server sceglie chi partecipa e propone un intervallo di lavoro; quanto
    lavorare lo decide il client, un istante prima di iniziare, con cio' che
    solo lui sa.

    ROUND 0 -- ISCRIZIONE (al primo round, nessun training)
      Il server registra i dati STATICI di ogni client (profilo hardware,
      batteria del device, taglia del dataset) e il SoC iniziale. Per tutti
      assume il carico medio della POPOLAZIONE, w_enroll (0.118 a x1): media
      delle utilizzazioni stazionarie delle tre catene di workload, un dato
      aggregato e non del singolo client. Fissa la DEADLINE del round:
      deadline_mult per la mediana dei tempi del lavoro standard (E e B delle
      baseline, senza carico). Vale 132 s e resta fissa per tutta la run.
      [B] col carico massimo, a beta 1.0 chi non era mai contattato veniva
      stimato scarico dal tempo 0, finiva in fondo alla classifica e non
      veniva piu' interrogato (6 client mai scelti a x1, 26 a x3). Un errore
      ottimista lo corregge la prima proposta; uno pessimista mai.
      [B] T_round limita lo straggler, quindi anche l'idle che gli altri
      pagano aspettandolo (modello ESCS). Senza deadline un client in carica
      riceveva le epoche massime e arrivava a 30 volte i passi degli altri.

    OGNI ROUND, IN SETTE PASSI

    1. PREVISIONE (stima del SoC lato server, nessun messaggio; _forecast)
       Per ogni client il server conserva l'ultimo stato ricevuto e quando.
       Prevede il SoC di adesso: scarica di idle con la batteria del device
       (dato statico) e probabilita' che nel frattempo il proprietario lo
       abbia messo in carica (modello orario degli utenti, Ferreira 2011; se
       era in carica, quanto si e' ricaricato). Prevede anche il workload,
       che torna dall'ultimo valore riportato verso la media del client.
       [B] CIRCOLARITA': batteria e consumo sono quelli del simulatore, quindi
       la previsione sbaglia solo per ricariche e workload. Su device veri
       sbaglierebbe anche per il modello: qui e' ottimista, va dichiarato.

    2. CLASSIFICA
           score = a*SoC_previsto + b*D + c*min(1, eta / stale_max)   a + b = 1
       D divergenza JS delle classi (come SAGE), eta i round dall'ultima
       selezione (chi non e' mai stato scelto conta come scelto al round 0),
       c = stale_weight. Il workload NON entra qui (vedi passo 5).
       [B] IL TERMINE DI ANZIANITA'. Con il solo a*SoC + b*D, D e' statica:
       chi ha divergenza bassa non viene scelto mai, e chi consuma poco resta
       carico e viene riscelto sempre (misurato: 4 client su 30 mai
       selezionati, uno scelto 228 volte su 250, Gini 0.62 contro 0.19 di
       FedAvg). L'anzianita' fa guadagnare punteggio a chi non partecipa da
       tanto, finche' non viene provato (Fase 8 della proposta, come Oort e
       F3AST).
       [B] 10/11: pesare la classifica per 1 - w atteso escludeva per sempre
       la catena di workload "high": il w atteso tende alla sua media e non
       scende mai, la penalita' colpiva anche l'anzianita' (catena high
       scelta 21 volte contro 40, 4 telefoni mai scelti, -1.5 punti).

    3. POSTI
       k al primo round, uno in piu' a ogni round, al massimo il 20% dei vivi
       (2 al round 1, 20 dal round 19 con 100 client). Con score non negativi
       e soli vincoli di cardinalita' l'ILP coincide con i top per score,
       come in SAGE.

    4. PROPOSTE (negoziazione; _negotiate_pick, _propose)
       Il modello non va ai primi della classifica: una PROPOSTA (l'envelope
       del passo 6, senza modello, MessageType.QUERY) va ai primi
       ceil(shortlist * posti), 30 a regime con shortlist = 1.5.
       - Senza broadcast nessuno ha verificato prima chi e' raggiungibile: la
         frazione broadcast_nonresp (10%) non riceve la proposta (fuori
         copertura, spento), non paga nulla e l'ondata si chiude al timeout.
       - Chi risponde invia lo stato di adesso (SoC, workload, in carica) e
         accetta o rifiuta. Rifiuta SOLO per energia: se il lavoro minimo
         (E_lo) lo porterebbe sotto la riserva; in carica accetta sempre
         (client_policy.accepts_proposal).
       - Un rifiuto conta come contatto: l'anzianita' riparte da zero, cosi'
         il client non torna subito in cima e non si spreca un messaggio a
         ogni round. Chi non risponde non e' contattato.
       - Se le accettazioni non coprono i posti, una seconda ondata va ai
         successivi della classifica (ceil(shortlist * posti mancanti)), al
         massimo max_waves ondate. Se nessuno accetta, il pool e' esaurito e
         la run termina, come quando l'ILP e' infeasible.
       COSTO: ogni scambio e' una sessione radio LTE (~12.6 J, vedi
       communication_model). Chi viene scelto la condivide col download del
       modello che segue, gli altri la pagano intera. Ogni ondata ritarda il
       round di promozione + RTT (~0.33 s), piu' il timeout se manca qualcuno.
       [B] l'ex broadcast dello stato a tutti i client non serve: le risposte
       alle proposte danno sia la raggiungibilita' sia lo stato (energia di
       controllo da ~10 a ~1 Wh per run).

    5. SCELTA FRESCA (_fresh_score)
       Fra chi ha accettato il server tiene i migliori, tanti quanti i posti,
       con lo stato APPENA riportato:
           score_fresco = (a*SoC_ora + b*D + c*anzianita') * (1 - w_ora)
       Un telefono in carica conta come SoC = 1 (il training non costa nulla
       alla sua batteria); uno occupato pesa meno (nella deadline farebbe
       meno passi). Gli altri che avevano accettato sono RINVIATI: tengono
       l'anzianita' e sono in cima alla classifica del round dopo.
       [B] il workload pesa solo qui, dove e' quello appena riportato e
       quindi transitorio.

    6. ENVELOPE E TRAINING LOCALE (_assign_work; client: client_policy.decide)
           E in [E_lo, E_hi]   B in {batch ammessi}   + deadline, riserva
       E_lo = epochs_min: sotto, l'update non porta segnale utile.
       E_hi = min(epochs_cap_mult * E, vincolo di deadline, vincolo di energia):
         deadline: E * t_epoca(B piu' veloce, w noto) <= T_round
         energia:  E * dSoC_epoca <= SoC_noto - soc_min - dSoC_upload, con dSoC
                   stimato LINEARMENTE (P*t/capacita') e corretto da un fattore
                   k_i appreso dai consumi che il client riporta (passo 7).
       B: le potenze di 2 fra batch_min e batch_max (16..128).
       Il client rilegge SoC, workload e carica, misura la loss del modello sui
       propri dati e sceglie E in proporzione a loss e batteria sopra la
       riserva, B dal piu' piccolo (piu' passi di SGD nello stesso tempo). Se
       nemmeno E_lo sta nella deadline fa i passi che ci stanno (lavoro
       parziale); se nemmeno E_lo rispetta la riserva si ritira (opt-out).
       Durante il training si ferma se finisce il budget energetico.
       [B] con la potenza costante per tier l'energia per CAMPIONE cala al
       crescere di B, quella per PASSO di SGD cresce: con la stessa energia
       B = 16 fa molti piu' passi di B = 128. Le baseline restano a
       batch_size = 128: parte del vantaggio di SAGE-smart in accuracy viene
       da B (ablazione: FedAvg con B piccolo, results_sweepB*).
       [B] il vincolo di energia e' volutamente LASCO: il server non conosce
       il workload attuale ne' la curva vera della batteria. Il compito di
       essere precisi e' del client.

    7. AGGREGAZIONE E AGGIORNAMENTI (aggregate_train, _on_round_result)
       Il client comunica (E, B), passi eseguiti, SoC iniziale e FINALE, loss
       prima e dopo, istogramma delle classi. Il server:
         - scarta gli update fuori envelope e addebita al mondo fisico il
           lavoro DICHIARATO (_work_done);
         - fa la media FedAvg pesata per i dati; nei round con lavoro
           parziale, pesata per i campioni visti davvero;
         - aggiorna stato noto (SoC finale riportato), k_i, divergenza D_i e
           anzianita' (a zero) di chi si e' allenato.

    [B] LATE BINDING: la negoziazione avviene PRIMA di WorldState.begin_round,
      che fa avanzare il workload e decide le ricariche: resta uno scarto fra
      cio' che il client ha dichiarato e cio' che trova al training. Per
      questo il client decide (E, B) al training, e puo' ancora ritirarsi.

    INTERRUTTORI DELLE ABLAZIONI (smart-ablation in server_app, "+" fra piu'
      componenti), tutti accesi di default:
        stale_weight 0      nostale    niente anzianita'
        forecast False      noest      SoC e workload dell'ultimo report
        negotiation False   noneg      niente proposte: il modello va ai primi
                                       della classifica (possono ancora
                                       ritirarsi); torna il broadcast dello
                                       stato per sapere chi e' raggiungibile
        e5, b128                       E = 5 o B = 128 fissi (server_app)
        partial_work False  nopartial  niente lavoro parziale

    COSA SA IL SERVER, E DA DOVE
      Le decisioni usano solo: profili statici e tipo di batteria (round 0),
      risposte alle proposte (passo 4), metriche riportate dai client (passo
      7). Il mondo fisico entra in due soli punti, fuori dalle decisioni:
        - `_device_record`: i SENSORI del device, allegati al messaggio per
          conto del device stesso (in un telefono vero li legge dall'OS);
        - le colonne smart_soc_err, smart_info_err e smart_w_err, che
          confrontano dichiarato/stimato e vero: sono verifiche, non input.

    PARAMETRI
      stale_weight, stale_max  peso (c) e scala dell'anzianita'
      soc_min        riserva che il CLIENT protegge (vincolo 2e di SAGE)
      epochs_min     E_lo dell'envelope
      epochs_cap_mult  E_hi al massimo epochs_cap_mult * E delle baseline
                     (2 * 5 = 10): al massimo il doppio del lavoro standard
      batch_min, batch_max  B dell'envelope: potenze di 2 fra i due
      deadline_mult  scala della deadline sulla mediana nominale
      probe_samples  campioni per stimare la loss prima di decidere
      max_waves      ondate di proposte per round (1 = nessun rimpiazzo)
      shortlist      proposte a shortlist * posti candidati (rosa)
      load_tau_s     costante con cui il workload previsto torna alla media
      query_timeout_s  attesa massima delle risposte ai messaggi di controllo
      calib_alpha    peso della media mobile su k_i
    """

    def __init__(
        self,
        *args,
        stale_weight: float = 0.3,
        stale_max: int = 25,
        soc_min: float = 0.20,
        epochs_cap_mult: int = 2,
        epochs_min: int = 2,
        partition_sizes: dict[int, int] | None = None,
        deadline_mult: float = 1.0,
        probe_samples: int = 256,
        max_waves: int = 2,
        query_timeout_s: float = 600.0,
        calib_alpha: float = 0.3,
        batch_min: int = 16,
        batch_max: int = 64,
        forecast: bool = True,
        negotiation: bool = True,
        partial_work: bool = True,
        shortlist: float = 1.5,
        load_tau_s: float = 450.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if shortlist < 1.0:
            raise ValueError(f"shortlist >= 1, non {shortlist}")
        self.forecast = bool(forecast)
        self.negotiation = bool(negotiation)
        self.partial_work = bool(partial_work)
        self.shortlist = float(shortlist)
        self.load_tau_s = float(load_tau_s)
        # [B] senza broadcast nessuno ha verificato chi e' raggiungibile: la
        # frazione broadcast_nonresp dei contattati non risponde (proposta o
        # modello). Il broadcast lo spegne server_app quando c'e' la
        # negoziazione; resta solo nell'ablazione senza negoziazione (noneg).
        self.proposal_nonresp = not self.broadcast
        self.stale_w = float(stale_weight)
        self.stale_max = int(stale_max)
        self.soc_min = float(soc_min)
        if not 1 <= int(batch_min) <= int(batch_max):
            raise ValueError(f"batch: serve 1 <= min <= max, non {batch_min}, {batch_max}")
        choices, b = [], int(batch_min)
        while b <= int(batch_max):
            choices.append(b)
            b *= 2
        self.batch_choices = tuple(choices)
        self.epochs_cap = int(epochs_cap_mult) * int(self.epochs)
        self.epochs_min = int(epochs_min)
        self.deadline_mult = float(deadline_mult)
        self.probe_samples = int(probe_samples)
        self.max_waves = max(1, int(max_waves))
        self.query_timeout_s = float(query_timeout_s)
        self.calib_alpha = float(calib_alpha)
        # Round 0: taglia dei dataset. Se manca, la si impara dalle risposte.
        self._n: dict[int, int] = {int(c): int(n) for c, n in
                                   (partition_sizes or {}).items()}
        # ripiego se una risposta non riportasse il workload: il caso peggiore
        # (segue workload-scale: a x1 e' 0.30)
        self.w_busy = self.world.workload.max_utilization()
        # carico assunto all'iscrizione: la media della POPOLAZIONE (vedi la
        # docstring della classe, round 0)
        wl = self.world.workload
        self.w_enroll = (float(np.mean(list(stationary_utilization(wl.scale).values())))
                         if wl.enabled else 0.0)

        self._last_seen: dict[int, int] = {}       # cid -> ultimo round scelto
        self._plan: dict[int, Envelope] = {}       # cid -> envelope del round

        # stato che il server tiene per ogni client: l'ultimo REPORT (cosa il
        # client ha detto e quando) e lo stato NOTO (previsione o report fresco)
        self._subscribed = False
        self._deadline_s: float = math.inf
        self._clock_s = 0.0                        # tempo simulato, inizio round
        self._soc_rep: dict[int, float] = {}
        self._t_rep: dict[int, float] = {}
        self._w_rep: dict[int, float] = {}
        self._w_sum: dict[int, float] = {}         # workload riportati, per la media
        self._w_cnt: dict[int, int] = {}
        self._chg_rep: dict[int, bool] = {}
        self._soc_known: dict[int, float] = {}
        self._chg_known: dict[int, bool] = {}
        self._w_known: dict[int, float] = {}
        self._k_cal: dict[int, float] = {}
        self._loss_known: dict[int, float] = {}
        # per round: vincolo che ha fissato E_hi, risposte dei client e costi
        # della negoziazione
        self._bound: dict[int, str] = {}
        self._reports: dict[int, dict] = {}
        self._grid = None
        self._queried: set[int] = set()
        self._overhead_s = 0.0
        self._waves = 0
        self._n_queried = 0
        self._n_refused = 0
        self._info_err = float("nan")
        # rinviati, non raggiunti, attesa del timeout per chi non risponde,
        # errore sul workload noto (verifica)
        self._n_deferred = 0
        self._n_noreply = 0
        self._extra_wait_s = 0.0
        self._w_err = float("nan")

    @property
    def _protocol_cols(self) -> bool:
        """Previsione, negoziazione o niente broadcast: colonne in piu' nel CSV
        (rinviati, non raggiunti, errore sul workload). Spente solo nei
        gradini delle cumulate senza stima e senza negoziazione, che cosi'
        hanno le colonne di prima."""
        return self.forecast or self.negotiation or self.proposal_nonresp

    @property
    def variant(self) -> str:
        off = [name for name, on in (
            ("anzianita'", self.stale_w > 0),
            ("previsione del SoC", self.forecast),
            ("negoziazione (broadcast dello stato)" if self.broadcast else "negoziazione",
             self.negotiation),
            ("lavoro parziale", self.partial_work)) if not on]
        return (f"sage-smart-{self.world.battery}"
                + (f" SENZA {', '.join(off)}" if off else ""))

    def configure_train(self, server_round, arrays, config, grid):
        # la griglia serve alla negoziazione per i messaggi di controllo, che
        # partono dentro _select_clients prima dei messaggi di training
        self._grid = grid
        return super().configure_train(server_round, arrays, config, grid)

    def _round_overhead_s(self) -> float:
        # negoziazione + broadcast dello stato (solo con noneg, vedi PhysicalFedAvg)
        return self._overhead_s + super()._round_overhead_s()

    # ========================================== round 0 e passo 1: lo stato
    def _sensor_soc(self, cid: int) -> float:
        """SoC che il device legge dalla propria batteria."""
        return float(self.world.snapshot(cid).soc)

    def _subscribe(self) -> None:
        """Iscrizione: dati statici e SoC iniziale di ogni client."""
        cids = list(self._node_to_cid.values())
        for c in cids:
            # il primo "report" e' l'iscrizione, al tempo 0, col workload medio
            # della popolazione (non entra nella media dei workload del client:
            # non e' una lettura vera)
            self._record(c, self._sensor_soc(c), self.w_enroll, False, 0.0,
                         measured=False)
            self._k_cal[c] = 1.0
        nominal = [
            training_time_s(self.world.profile(c), int(self.epochs),
                            self._n[c], int(self.batch_size))
            for c in cids if self._n.get(c)
        ]
        if nominal:
            self._deadline_s = self.deadline_mult * float(np.median(nominal))
        self._subscribed = True
        log(INFO, "%s: %s client iscritti | deadline %.1f s "
            "(%.2f x mediana nominale, E=%s B=%s) | max %s ondate | batteria %s",
            self.variant, len(cids), self._deadline_s, self.deadline_mult,
            self.epochs, self.batch_size, self.max_waves,
            BATTERY_NAMES[self.world.battery])

    def _record(self, cid: int, soc: float, w: float, chg: bool, t: float,
                measured: bool = True) -> None:
        """Un report del client: diventa anche lo stato noto."""
        if measured:
            self._w_sum[cid] = self._w_sum.get(cid, 0.0) + float(w)
            self._w_cnt[cid] = self._w_cnt.get(cid, 0) + 1
        self._soc_rep[cid], self._w_rep[cid] = float(soc), float(w)
        self._chg_rep[cid], self._t_rep[cid] = bool(chg), float(t)
        self._soc_known[cid], self._w_known[cid] = float(soc), float(w)
        self._chg_known[cid] = bool(chg)

    def _forecast(self, cid: int) -> tuple[float, float]:
        """Passo 1: SoC e workload ATTESI adesso.

            w(t)   = w_med + (w_rep - w_med) * exp(-(t - t_rep) / tau)
            SoC(t) = valore atteso su tre rami, a passi di _ESTIMATE_STEP_S:
                     in scarica (idle con w(t)), in carica (C-rate fino al
                     100%), carica finita e di nuovo in scarica.

        w_med e' la media dei workload riportati dal client (prima del primo
        report, quella della popolazione). Dal ramo in scarica passa in carica,
        a ogni passo, la frazione p = 1 - exp(-lambda (1 - SoC) dt): e' il
        modello orario degli utenti (Ferreira 2011), un dato di POPOLAZIONE
        che il server conosce come conosce la batteria del device. Un client
        in carica all'ultimo report resta in carica fino al 100%. L'idle e'
        integrato a passi di al piu' _ESTIMATE_STEP_S, come lo scarica il mondo
        round per round: con la batteria da datasheet il calo dipende dal SoC.

        [B] PERCHE'. Una stima che scarica sempre e non ricarica mai fa
        scendere in classifica un client non contattato da ore anche se nel
        frattempo, in media, ha ricaricato (+2 punti l'ora a SoC 0.6). E un
        workload fermo all'ultimo valore sbaglia: dopo 5 round lo stato e' lo
        stesso solo nel 51% dei casi.
        """
        soc0 = float(self._soc_rep[cid])
        dt = max(0.0, self._clock_s - self._t_rep.get(cid, 0.0))
        w_rep = float(self._w_rep.get(cid, self.w_enroll))
        w_med = (self._w_sum[cid] / self._w_cnt[cid]) if self._w_cnt.get(cid) else self.w_enroll
        if dt <= 0.0:
            return soc0, w_rep
        prof = self.world.profile(cid)
        bat = self.world.battery
        recharge = bool(getattr(self.world, "_recharge_enabled", False))
        n = max(1, math.ceil(dt / _ESTIMATE_STEP_S))
        h = dt / n
        # masse e SoC dei tre rami: scarica (d), in carica (c), carica finita (f)
        if self._chg_rep.get(cid, False):
            m_d, s_d, m_c, s_c = 0.0, soc0, 1.0, soc0
        else:
            m_d, s_d, m_c, s_c = 1.0, soc0, 0.0, soc0
        m_f, s_f = 0.0, 1.0
        for i in range(n):
            w = w_med + (w_rep - w_med) * math.exp(-(i + 0.5) * h / self.load_tau_s)
            p_idle = idle_power_w(prof, w)
            if recharge and m_d > 0.0:
                moved = m_d * (1.0 - math.exp(-RECHARGE_RATE_PER_H * (1.0 - s_d) * h / 3600.0))
                if moved > 0.0:
                    s_c = (m_c * s_c + moved * s_d) / (m_c + moved)
                    m_c, m_d = m_c + moved, m_d - moved
            if m_d > 0.0:
                s_d = max(0.0, s_d + battery_delta_soc(bat, p_idle, h, prof, s_d))
            if m_f > 1e-9:
                s_f = max(0.0, s_f + battery_delta_soc(bat, p_idle, h, prof, s_f))
            if m_c > 0.0:
                s_c = min(1.0, s_c + CHARGE_C_RATE * h / 3600.0)
                if s_c >= 1.0:
                    s_f = (m_f * s_f + m_c) / (m_f + m_c)
                    m_f, m_c = m_f + m_c, 0.0
        soc = m_d * s_d + m_c * s_c + m_f * s_f
        w_now = w_med + (w_rep - w_med) * math.exp(-dt / self.load_tau_s)
        return float(min(1.0, max(0.0, soc))), float(w_now)

    def _refresh_state(self, available: list[int]) -> None:
        """Passo 1: lo stato con cui il server ordina i client."""
        for c in self._node_to_cid.values():
            if self.forecast:
                self._soc_known[c], self._w_known[c] = self._forecast(c)
            else:
                # ablazione noest: l'ultimo report cosi' com'e', anche se
                # vecchio di molti round
                self._soc_known[c] = self._soc_rep[c]
                self._w_known[c] = self._w_rep[c]
            self._chg_known[c] = self._chg_rep[c]
        # [B] verifica, non input: quanto lo stato noto si discosta da quello
        # che il device leggerebbe adesso, sul pool selezionabile
        pool = [self._node_to_cid[n] for n in available]
        if pool:
            self._info_err = float(np.mean(
                [abs(self._soc_known[c] - self._sensor_soc(c)) for c in pool]))
            if self._protocol_cols:
                self._w_err = float(np.mean(
                    [abs(self._w_known[c] - self.world.workload.utilization(c))
                     for c in pool]))

    def _propose(self, nodes: list[int],
                 envs: dict[int, Envelope]) -> dict[int, Message]:
        """Passo 4: proposte (MessageType.QUERY) e lettura delle risposte.

        Le risposte aggiornano subito lo stato noto. Il costo energetico NON
        si addebita qui: dipende da chi verra' selezionato (sessione radio
        condivisa col download del modello o no), vedi _charge_control.
        """
        if self.proposal_nonresp:
            # [B] senza broadcast nessuno ha verificato prima chi e'
            # raggiungibile. La frazione broadcast_nonresp non riceve la
            # proposta (fuori copertura, spento): non paga nulla, non risponde,
            # e l'ondata si chiude al timeout.
            lost = {n for n in nodes if self._bcast_rng.random() < self.bcast_nonresp}
            if lost:
                self._n_noreply += len(lost)
                self._extra_wait_s += max(0.0, self.bcast_timeout_s - ctrl_latency_s())
                nodes = [n for n in nodes if n not in lost]
        msgs: list[Message] = []
        for n in nodes:
            cid = self._node_to_cid[n]
            cfg = ConfigRecord({"cid": int(cid)})
            for key, val in envs[cid].to_config().items():
                cfg[key] = val
            rec = RecordDict({self.configrecord_key: cfg,
                              "device": self._device_record(cid)})
            msgs.extend(self._construct_messages(rec, [int(n)], MessageType.QUERY))
        self._queried.update(nodes)
        self._n_queried += len(nodes)
        if self._grid is None or not msgs:
            return {}
        out: dict[int, Message] = {}
        for r in self._grid.send_and_receive(msgs, timeout=self.query_timeout_s):
            if not r.has_content():
                continue
            n = int(r.metadata.src_node_id)
            cid = self._node_to_cid.get(n)
            if cid is None:
                continue
            self._record(cid, float(self._rep(r, "smart-soc", self._soc_known[cid])),
                         float(self._rep(r, "smart-workload", self.w_busy)),
                         bool(int(self._rep(r, "smart-charging", 0))),
                         self._clock_s)
            out[n] = r
        return out

    def _charge_control(self, selected: list[int]) -> None:
        """Addebita al mondo gli scambi della negoziazione del round.

        Chi e' stato selezionato riceve subito dopo il modello: lo scambio
        cade nella stessa sessione radio (merged). Chi non lo e' paga una
        sessione intera.
        """
        sel = set(selected)
        for n in self._queried:
            # [B] col broadcast attivo ogni candidato ha appena risposto alla
            # richiesta di stato: la sessione radio e' gia' aperta e la
            # proposta paga il solo payload, anche se viene rifiutata
            self.world.apply_control(self._node_to_cid[n],
                                     merged=(n in sel) or self.broadcast)

    def _resource(self, cid: int) -> float:
        """SoC usato nello score: quello NOTO al server, non quello vero."""
        return self._soc_known.get(cid, float(self.world.initial_soc(cid)))

    def _n_of(self, cid: int) -> int | None:
        if cid in self._n:
            return self._n[cid]
        return int(np.median(list(self._n.values()))) if self._n else None

    def _batch_for(self, cid: int) -> int:
        """B preferito: il piu' piccolo dell'envelope (passo 6 nella docstring
        della classe). Il B effettivo lo sceglie il client e lo dichiara."""
        return int(self.batch_choices[0])

    def _staleness(self, cid: int, server_round: int) -> float:
        """Round dall'ultima selezione, normalizzati su stale_max.

        Chi non e' mai stato scelto conta come scelto al round 0: la sua
        anzianita' cresce dal primo round come quella degli altri, e lo porta
        in classifica invece di escluderlo per sempre.
        """
        last = self._last_seen.get(cid)
        age = server_round if last is None else server_round - last
        return float(min(1.0, age / max(1, self.stale_max)))

    # ========================================================= passi 2-5
    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not self._subscribed:
            self._subscribe()
        self._reports = {}
        self._bound = {}
        self._queried = set()
        self._overhead_s = 0.0
        self._waves = 0
        self._n_queried = 0
        self._n_refused = 0
        self._n_deferred = 0
        self._n_noreply = 0
        self._extra_wait_s = 0.0
        if not k or not available:
            log(INFO, "%s round %s: pool vuoto", self.variant, server_round)
            return []
        self._refresh_state(available)
        cid_of = self._node_to_cid
        pool = available
        m = len(pool)
        # tetto sui VIVI, non sui soli raggiunti dal broadcast (vedi
        # PhysicalSAGEAblation._select_clients)
        hi = min(self.k + server_round - 1,
                 int(np.floor(self.max_frac * max(m, self._n_alive))), m)
        lo = self.k

        score = {
            n: self.a * self._resource(cid_of[n])
            + self.b * self._divergence(cid_of[n])
            + self.stale_w * self._staleness(cid_of[n], server_round)
            for n in pool
        }

        prob = pulp.LpProblem("SAGE_Smart_Select", pulp.LpMaximize)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]
        prob += pulp.lpSum(y[i] * score[pool[i]] for i in range(m))
        prob += pulp.lpSum(y) >= lo
        prob += pulp.lpSum(y) <= hi
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [pool[i] for i in range(m) if int(pulp.value(y[i]) or 0) == 1]
        if not selected:
            self._charge_control([])
            log(INFO, "%s round %s: ILP infeasible con pool %s (lo=%s, hi=%s): "
                "pool esaurito", self.variant, server_round, m, lo, hi)
            return []

        # [B] l'ILP con soli vincoli di cardinalita' sceglie i top per score:
        # le proposte scorrono la stessa classifica
        ranking = sorted(pool, key=lambda n: -score[n])
        if self.negotiation:
            selected = self._negotiate_pick(ranking, len(selected), server_round)
        else:
            # ablazione noneg: i primi della classifica senza chiedere, nessun
            # rimpiazzo e nessuno scambio di controllo (waves = 0, overhead 0)
            selected = ranking[:len(selected)]
            if self.proposal_nonresp:
                # [B] senza broadcast e senza proposte: chi non e'
                # raggiungibile non riceve il modello e il suo posto resta
                # vuoto (il round dura comunque fino alla deadline)
                lost = {n for n in selected
                        if self._bcast_rng.random() < self.bcast_nonresp}
                self._n_noreply += len(lost)
                selected = [n for n in selected if n not in lost]
        if not selected:
            self._charge_control([])
            log(INFO, "%s round %s: nessun candidato ha accettato in %s "
                "ondate: pool esaurito", self.variant, server_round,
                self._waves)
            return []

        self._charge_control(selected)
        self._overhead_s = self._waves * ctrl_latency_s() + self._extra_wait_s
        for n in selected:
            self._last_seen[cid_of[n]] = server_round
        # envelope con lo stato appena riportato dai client che hanno accettato
        self._plan = {cid_of[n]: self._assign_work(cid_of[n]) for n in selected}

        E = [env.e_hi for env in self._plan.values()]
        never = sum(1 for c in cid_of.values() if c not in self._last_seen)
        log(INFO, "%s round %s: pool %s, %s selezionati (cap %s) | "
            "E_hi %s-%s (medio %.1f) | mai visti %s | interrogati %s, "
            "rifiuti %s, ondate %s%s",
            self.variant, server_round, m, len(selected), hi, min(E), max(E),
            float(np.mean(E)), never, self._n_queried, self._n_refused,
            self._waves,
            (f" | rinviati {self._n_deferred}, non raggiunti {self._n_noreply}, "
             f"errore w {self._w_err:.3f}") if self._protocol_cols else "")
        return selected

    def _fresh_score(self, cid: int, server_round: int) -> float:
        """Passo 5: lo score con lo stato APPENA riportato dal client.

        Stessa forma della classifica, ma con SoC, carica e workload della
        risposta alla proposta: un telefono in carica conta come pieno (la
        batteria non paga), e il punteggio e' pesato per 1 - w, il lavoro
        utile che puo' fare adesso.
        """
        soc = 1.0 if self._chg_rep.get(cid, False) else float(self._soc_rep[cid])
        base = (self.a * soc + self.b * self._divergence(cid)
                + self.stale_w * self._staleness(cid, server_round))
        return base * (1.0 - float(self._w_rep.get(cid, self.w_enroll)))

    def _negotiate_pick(self, ranking: list[int], slots: int,
                        server_round: int) -> list[int]:
        """Passi 4-5: rosa piu' larga dei posti, scelta sullo stato fresco.

        Ondata 1: proposte ai primi ceil(shortlist * slots) della classifica.
        Chi accetta entra nella rosa; chi rifiuta (solo per energia) perde
        l'anzianita'; chi non risponde non e' raggiungibile e non conta come
        contatto. Se i posti non sono coperti, ondata 2 coi successivi, ancora
        shortlist volte quelli che mancano. Poi il server sceglie i `slots`
        migliori per _fresh_score: gli altri della rosa sono RINVIATI, tengono
        l'anzianita' e restano in cima al round dopo.
        """
        cid_of = self._node_to_cid
        accepted: list[int] = []
        asked: set[int] = set()
        need = slots
        while need > 0 and self._waves < self.max_waves:
            batch = [n for n in ranking if n not in asked][: int(math.ceil(self.shortlist * need))]
            if not batch:
                break
            self._waves += 1
            asked.update(batch)
            envs = {cid_of[n]: self._assign_work(cid_of[n]) for n in batch}
            replies = self._propose(batch, envs)
            for n in batch:
                r = replies.get(n)
                if r is None:
                    continue
                if int(self._rep(r, "smart-accept", 0)) == 1:
                    accepted.append(n)
                else:
                    self._n_refused += 1
                    # [B] contattato: l'anzianita' riparte da zero, cosi' non
                    # torna in testa alla classifica al round successivo
                    self._last_seen[cid_of[n]] = server_round
            need = slots - len(accepted)
        picked = sorted(accepted,
                        key=lambda n: -self._fresh_score(cid_of[n], server_round))[:slots]
        self._n_deferred = len(accepted) - len(picked)
        return picked

    def _mark_half(self, selected: list[int]) -> None:
        """Disattivata: le mezze epoche di SAGE qui non servono, la riduzione
        del lavoro per batteria scarica la decide il client."""
        self._half = set()

    # ============================================================ passo 6
    def _batches_for(self, cid: int) -> tuple[int, ...]:
        """Taglie di batch dell'envelope, dalla piu' piccola (preferita)."""
        return self.batch_choices

    def _assign_work(self, cid: int) -> Envelope:
        prof = self.world.profile(cid)          # statico: dal round 0
        batches = self._batches_for(cid)
        b_ref = self._batch_for(cid)
        peff = effective_profile(prof, self._w_known.get(cid, self.w_enroll))
        e_lo, e_cap = int(self.epochs_min), int(self.epochs_cap)
        n = self._n_of(cid)

        # vincolo di deadline: col batch piu' veloce dell'envelope
        e_dl = e_cap
        if n and math.isfinite(self._deadline_s):
            t_ep = min(training_time_s(peff, 1, n, b) for b in batches)
            e_dl = int(self._deadline_s // max(t_ep, 1e-9))

        # vincolo di energia: stima lineare corretta da k_i
        e_en = e_cap
        if n and not self._chg_known.get(cid, False):
            cap_wh = prof.battery_capacity_mah / 1000.0 * V_NOMINAL
            k = self._k_cal.get(cid, 1.0)
            d_ep = k * energy_wh_linear(
                training_power_w(prof), training_time_s(peff, 1, n, b_ref)
            ) / cap_wh
            d_up = k * energy_wh_linear(comm_power_w(), comm_time_s()) / cap_wh
            spare = self._resource(cid) - self.soc_min - d_up
            e_en = int(spare // d_ep) if d_ep > 0 else e_cap

        e_hi = int(np.clip(min(e_cap, e_dl, e_en), e_lo, e_cap))
        if e_hi >= e_cap:
            self._bound[cid] = "cap"
        elif e_dl <= e_en:
            self._bound[cid] = "deadline"
        else:
            self._bound[cid] = "energy"
        return Envelope(
            e_lo=e_lo, e_hi=e_hi, batches=batches,
            deadline_s=float(self._deadline_s) if math.isfinite(self._deadline_s)
            else 1e12,
            soc_min=float(self.soc_min), probe_samples=self.probe_samples,
            partial_work=self.partial_work,
        )

    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        env = self._plan.get(cid)
        if env is None:
            return
        for key, val in env.to_config().items():
            cfg[key] = val
        # [B] epochs e batch-size restano nel config come TETTO e batch di
        # riferimento: il client di sage_smart non li usa per allenarsi.
        cfg["epochs"] = int(env.e_hi)
        cfg["batch-size"] = int(self._batch_for(cid))

    def _device_record(self, cid: int) -> ConfigRecord | None:
        """Sensori del device: letti dal mondo PER CONTO del client.

        [B] nel messaggio di training e' chiamato dopo WorldState.begin_round:
        il client vede il workload e lo stato di carica di QUESTO round, il
        server nella selezione aveva visto quelli del precedente.
        [B] "dataset-size": il client conosce la taglia del proprio dataset;
        nel simulatore arriva qui per non dover caricare i dati in una query.
        """
        rec = DeviceReading.from_world(self.world, cid).to_record()
        rec["dataset-size"] = int(self._n_of(cid) or 0)
        return ConfigRecord(rec)

    # ============================================================ passo 7
    def _rep(self, reply: Message, key: str, default=None):
        return self._metric(reply, key, default=default)

    def _is_optout(self, reply: Message) -> bool:
        return int(self._rep(reply, "smart-optout", 0) or 0) == 1

    def _charge_communication(self, cid: int, reply: Message) -> tuple[float, float]:
        """Chi si ritira ha scaricato il modello ma non carica l'update: paga
        il download e la risposta di ritiro (uno scambio di controllo
        isolato), non lo scambio intero. [B] 04/11: prima pagava ~181 J invece
        di ~48 J, e l'ablazione senza negoziazione risultava penalizzata."""
        if not self._is_optout(reply):
            return super()._charge_communication(cid, reply)
        e_dl, l_dl = self.world.apply_download(cid)
        e_ct, l_ct = self.world.apply_control(cid, merged=False)
        return e_dl + e_ct, l_dl + l_ct

    def _work_done(self, cid: int, n_ex: int, reply: Message) -> tuple[float, int]:
        """Lavoro dichiarato dal client -> durata, per l'addebito fisico.

        Stessa formula che il client usa per prevedere il proprio SoC
        (DeviceReading.work_time_s).
        """
        b = int(self._rep(reply, "smart-batch", self._batch_for(cid)))
        steps = int(self._rep(reply, "smart-steps", 0))
        probe = int(self._rep(reply, "smart-probe", 0))
        e_eff = steps / math.ceil(n_ex / b) if steps > 0 and n_ex > 0 else 0.0
        dt = (self.world.round_duration_s(cid, epochs=e_eff, batch_size=b,
                                          dataset_size_local=n_ex)
              + self.world.probe_duration_s(cid, probe))
        return dt, b

    def _in_envelope(self, cid: int, reply: Message) -> bool:
        env = self._plan.get(cid)
        if env is None:
            return False
        e = int(self._rep(reply, "smart-epochs", 0))
        b = int(self._rep(reply, "smart-batch", 0))
        return env.e_lo <= e <= env.e_hi and b in env.batches

    def _aggregatable(self, replies: list[Message]) -> list[Message]:
        """Esclude chi si e' ritirato e chi e' uscito dall'envelope.

        [B] il client non dovrebbe mai uscire dall'envelope: la sua policy
        sceglie dentro i limiti. Il controllo rende esplicito il contratto:
        fuori dai limiti il server non accetta l'update (ma il lavoro e'
        stato fatto, quindi l'energia viene addebitata lo stesso).
        """
        out = []
        for r in replies:
            if not r.has_content():
                out.append(r)      # gli errori li gestisce FedAvg, come prima
                continue
            if self._is_optout(r):
                continue
            cid = self._node_to_cid.get(int(r.metadata.src_node_id))
            if cid is not None and not self._in_envelope(cid, r):
                continue
            out.append(r)
        return out

    def _steps(self, reply: Message) -> float:
        """Passi di SGD dichiarati dal client (colonne tau_* del CSV)."""
        if self._is_optout(reply):
            return 0.0
        return float(self._rep(reply, "smart-steps", 0))

    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        super()._on_round_result(cid, n_ex, dt_s, e_wh, e_lin_wh, reply,
                                 server_round)
        optout = self._is_optout(reply)
        soc0 = float(self._rep(reply, "smart-soc-start", self._resource(cid)))
        soc1 = float(self._rep(reply, "smart-soc-final", soc0))
        chg = bool(int(self._rep(reply, "smart-charging", 0)))
        w = float(self._rep(reply, "smart-workload", self.w_busy))
        b = int(self._rep(reply, "smart-batch", self._batch_for(cid)))
        steps = int(self._rep(reply, "smart-steps", 0))
        probe = int(self._rep(reply, "smart-probe", 0))

        # report del client, con l'istante in cui e' stato fatto: fine del
        # suo training e upload, dopo il ritardo iniziale del round. Da qui
        # partono le previsioni dei round successivi (passo 1).
        self._record(cid, soc1, w, chg, self._clock_s + self._overhead_s + dt_s)
        if n_ex > 0:
            self._n[cid] = int(n_ex)
        loss = self._rep(reply, "smart-loss-pre", None)
        if loss is not None:
            self._loss_known[cid] = float(loss)

        # calibrazione k_i: calo di SoC dichiarato / stima lineare dello
        # stesso lavoro, con la durata ricostruita dal workload DICHIARATO.
        # [B] nel mondo lineare k_i resta ~1; con Peukert e datasheet impara
        # quanto la batteria del device paga il training rispetto a P*t.
        if not chg and steps > 0 and n_ex > 0:
            prof = self.world.profile(cid)
            peff = effective_profile(prof, w)
            e_eff = steps / math.ceil(n_ex / b)
            t = training_time_s(peff, e_eff, n_ex, b) + probe_time_s(peff, probe)
            cap_wh = prof.battery_capacity_mah / 1000.0 * V_NOMINAL
            lin = (energy_wh_linear(training_power_w(prof), t)
                   + energy_wh_linear(comm_power_w(), comm_time_s())) / cap_wh
            drop = soc0 - soc1
            if lin > 0 and drop > 0:
                a = self.calib_alpha
                self._k_cal[cid] = (1 - a) * self._k_cal.get(cid, 1.0) + a * drop / lin

        self._reports[cid] = {
            "optout": optout,
            "epochs": int(self._rep(reply, "smart-epochs", 0)),
            "e_eff": (steps / math.ceil(n_ex / b)) if steps > 0 and n_ex > 0 else 0.0,
            "batch": b,
            "steps": steps,
            "budget_hit": int(self._rep(reply, "smart-budget-hit", 0)),
            "deadline_ok": int(self._rep(reply, "smart-deadline-ok", 1)),
            "charging": chg,
            "loss_pre": float(loss) if loss is not None else float("nan"),
            "loss_post": float(self._rep(reply, "smart-loss-post", float("nan"))),
            "workload": w,
            # [B] VERIFICA, non input: SoC dichiarato contro SoC VERO del
            # mondo dopo l'addebito di training + upload (l'idle viene dopo)
            "soc_err": abs(float(self.world.snapshot(cid).soc) - soc1),
            "rejected": (not optout) and not self._in_envelope(cid, reply),
            "partial": int(self._rep(reply, "smart-partial", 0)),
        }
        if self._reports[cid]["rejected"]:
            log(INFO, "%s round %s: update del client %s fuori envelope, "
                "scartato", self.variant, server_round, cid)

    def _policy_stats(self) -> dict:
        """Colonne della policy nel CSV per round."""
        rep = list(self._reports.values())
        done = [r for r in rep if not r["optout"]]
        envs = list(self._plan.values())

        def mean(xs):
            xs = [x for x in xs if x == x]          # scarta i NaN
            return float(np.mean(xs)) if xs else float("nan")

        st = {
            # epoche e batch ESEGUITI dai client
            "mean_epochs": mean([r["e_eff"] for r in done]),
            "max_epochs": float(max((r["e_eff"] for r in done), default=0.0)),
            "mean_batch": mean([r["batch"] for r in done]),
            "n_charging_sel": int(sum(1 for r in rep if r["charging"])),
            "smart_min_epochs": float(min((r["e_eff"] for r in done), default=0.0)),
            "smart_mean_e_hi": mean([e.e_hi for e in envs]),
            "smart_n_bound_energy": int(sum(1 for v in self._bound.values() if v == "energy")),
            "smart_n_bound_deadline": int(sum(1 for v in self._bound.values() if v == "deadline")),
            "smart_n_optout": int(sum(1 for r in rep if r["optout"])),
            "smart_n_rejected": int(sum(1 for r in rep if r["rejected"])),
            "smart_n_deadline_miss": int(sum(1 for r in done if not r["deadline_ok"])),
            "smart_n_budget_hit": int(sum(1 for r in done if r["budget_hit"])),
            "smart_mean_loss_pre": mean([r["loss_pre"] for r in rep]),
            "smart_mean_loss_post": mean([r["loss_post"] for r in done]),
            "smart_mean_workload": mean([r["workload"] for r in rep]),
            "smart_soc_err": float(max((r["soc_err"] for r in rep), default=0.0)),
            "smart_mean_k": mean(list(self._k_cal.values())),
            "smart_deadline_s": float(self._deadline_s),
            # passi 1 e 4: qualita' e costo dello stato usato per selezionare
            "smart_info_err": float(self._info_err),
            "smart_n_queried": int(self._n_queried),
            "smart_n_refused": int(self._n_refused),
            "smart_waves": int(self._waves),
            "smart_overhead_s": float(self._overhead_s),
        }
        if self.partial_work:
            # [B] solo con il lavoro parziale (nopartial ha le colonne di prima)
            st["smart_n_partial"] = int(sum(1 for r in done if r.get("partial")))
        if self._protocol_cols:
            # [B] rinviati (accettato ma non scelto), non raggiunti senza
            # broadcast, errore medio del workload noto (verifica)
            st["smart_n_deferred"] = int(self._n_deferred)
            st["smart_n_noreply"] = int(self._n_noreply)
            st["smart_w_err"] = float(self._w_err)
        log(INFO, "%s: %s risposte | E eseguite %.1f (min %.2f, max %.2f) "
            "| E_hi medio %.1f (energia %s, deadline %s) | opt-out %s | "
            "fuori deadline %s | errore SoC %.1e | errore stato noto %.3f",
            self.variant, len(rep), st["mean_epochs"], st["smart_min_epochs"],
            st["max_epochs"], st["smart_mean_e_hi"], st["smart_n_bound_energy"],
            st["smart_n_bound_deadline"], st["smart_n_optout"],
            st["smart_n_deadline_miss"], st["smart_soc_err"],
            st["smart_info_err"])
        return st

    # ------------------------------------------------------- aggregazione
    def aggregate_train(self, server_round, replies):
        """Media FedAvg pesata per i dati; col lavoro parziale, per i campioni
        visti davvero.

        La media per i dati la fa super() (PhysicalFedAvg) sulle reply di
        _aggregatable (senza opt-out e fuori envelope), BatchNorm compresa.
        Qui si aggiungono:
          - le colonne della policy nel CSV (_policy_stats);
          - le colonne tau_* coi passi di SGD dichiarati dai client
            (tau_min, tau_max, tau_eff = media pesata per i dati, tau_spread
            = max/min): solo diagnostica, nessun effetto sulla media;
          - [B] 06/11 LAVORO PARZIALE: nei round in cui qualcuno ha fatto meno
            di un'epoca, la media e' pesata per i campioni visti davvero,
            min(n, passi * B), invece che per n: un update da meta' dati non
            deve pesare come uno da tutti. Negli altri round resta la media di
            super(), identica.
        [B] 16/11: FedNova (normalizzazione per i passi, Wang et al. NeurIPS
        2020) tolto. Era spento dal 30/10: contributo esterno, e nelle due
        ablazioni la media pesata per i dati andava uguale o meglio (FedNova
        -0.9 / -1.0 punti).
        """
        arrays, metrics = super().aggregate_train(server_round, replies)
        # il server sa quanto e' durato il round: ha aspettato le risposte
        self._clock_s += self._last_round_s

        # [B] metriche della policy nel CSV per round, PRIMA dell'uscita
        # anticipata: servono anche quando l'aggregazione non viene applicata.
        # super() ha gia' assegnato _round_stats[round], quindi qui si aggiunge
        # sopra con update() invece di sovrascrivere.
        if self._plan:
            self._round_stats.setdefault(int(server_round), {}).update(
                self._policy_stats())

        if arrays is None or self._global_arrays is None:
            return arrays, metrics

        # [B] le CHIAVI vanno conservate: costruire un ArrayRecord da una lista
        # gli fa assegnare nomi posizionali "0","1",... e il load_state_dict in
        # global_evaluate non li riconosce. Leggere per chiave garantisce anche
        # lo stesso ordine fra record globale e locali.
        keys = list(self._global_arrays.keys())
        rows = []
        wts = []        # campioni visti davvero (smart-weight), per il lavoro parziale
        # stesse reply che super() ha mediato: senza opt-out e fuori envelope
        for r in self._aggregatable(replies):
            if not r.has_content():
                continue
            if self._node_to_cid.get(int(r.metadata.src_node_id)) is None:
                continue
            n = float(self._metric(r, "num-examples", "num_examples", default=0))
            tau = self._steps(r)
            if n <= 0 or tau <= 0:
                continue
            key = sorted(r.content.array_records)[0]
            rec = r.content.array_records[key]
            rows.append((n, tau, [rec[k].numpy() for k in keys]))
            wts.append(float(self._metric(r, "smart-weight", default=n)))

        if len(rows) < 2:
            return arrays, metrics

        tot = sum(n for n, _, _ in rows)
        p = [n / tot for n, _, _ in rows]
        tau_eff = sum(pi * tau for pi, (_, tau, _) in zip(p, rows))
        taus = [tau for _, tau, _ in rows]
        self._round_stats.setdefault(int(server_round), {}).update({
            "tau_min": int(min(taus)),
            "tau_max": int(max(taus)),
            "tau_eff": float(tau_eff),
            "tau_spread": float(max(taus) / max(1.0, min(taus))),
        })
        log(INFO, "round %s: FedAvg su %s client | tau %s-%s, tau_eff %.1f, "
            "spread %.1f", server_round, len(rows), int(min(taus)),
            int(max(taus)), tau_eff, max(taus) / max(1.0, min(taus)))

        if self.partial_work and any(w < n for w, (n, _, _) in zip(wts, rows)):
            tot_w = sum(wts)
            q = [w / tot_w for w in wts]
            merged = ArrayRecord()
            for j, k in enumerate(keys):
                ref = self._global_arrays[k].numpy()
                avg = np.asarray(sum(qi * row[2][j].astype(np.float64) for qi, row in zip(q, rows)),
                                 dtype=np.float64).reshape(ref.shape)
                if np.issubdtype(ref.dtype, np.integer):
                    avg = np.rint(avg)
                merged[k] = Array(avg.astype(ref.dtype))
            self._round_stats.setdefault(int(server_round), {}).update({
                "smart_weight_frac": float(tot_w / sum(n for n, _, _ in rows)),
            })
            return merged, metrics

        return arrays, metrics
