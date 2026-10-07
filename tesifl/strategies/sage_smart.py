from __future__ import annotations

import math
from logging import INFO

import numpy as np
import pulp
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MessageType, RecordDict
from flwr.common import log

from device.battery_model import energy_wh_linear
from device.communication_model import comm_power_w, comm_time_s
from device.compute_model import optimal_batch_size, probe_time_s, training_time_s
from device.communication_model import ctrl_latency_s
from device.constants import IDLE_POWER_W, TRAIN_POWER_W, V_NOMINAL, WORKLOAD_STATES
from device.fuel_gauge import DeviceReading
from device.world_state import effective_profile, training_power_w
from strategies.sage_ablation import PhysicalSAGEAblation
from tesiFL.client_policy import Envelope

# battery_mode (stessi nomi di ESCS) -> modello del fuel gauge del device
BATTERY_MODELS = {"soc": "nm", "peuk": "peuk", "energy": "lin"}
MODEL_NAMES = {"nm": "datasheet", "peuk": "Peukert", "lin": "lineare"}


class PhysicalSAGESmart(PhysicalSAGEAblation):
    """SAGE-smart: selezione lato server, (E, B) decisi dal CLIENT.

    Il server sceglie chi partecipa e propone un intervallo di lavoro; quanto
    lavorare lo decide il client, un istante prima di iniziare, con cio' che
    solo lui sa. Le fasi sono quelle della proposta (Sez. 4).

    FASE 0 -- ISCRIZIONE (una volta, al primo round)
      Il server registra i dati STATICI di ogni client: profilo hardware
      (capacita', velocita', curva di potenza), taglia del dataset, SoC
      iniziale. Ne ricava la deadline del round e il batch ottimo per device.

    FASE 2 -- STATO DEI CLIENT E NEGOZIAZIONE (documento "Client Server
              Protocol" del relatore, opzione 3 con l'opzione 2)
      a) STIMA LATO SERVER (opzione 2), nessun messaggio. Dall'ultimo report
         di ogni client il server estrapola il SoC col consumo di idle:
             SoC_est = SoC_rep - P_idle(w_rep) * dt / (V_nom * C_nom)
         con P_idle(w) = P_idle + w (P_train - P_idle) dal profilo statico,
         w_rep l'ultimo workload riportato, dt il tempo simulato dal report.
         Con questa stima ordina i client (Fase 3). Non vede ricariche ne'
         cambi di workload.
         [B] CIRCOLARITA': la formula ha la stessa forma del modello di idle
         del simulatore, quindi sbaglia solo per ricariche e workload che
         cambia. Su device veri sbaglierebbe anche per il modello: qui e'
         ottimista, va dichiarato.
      b) PROPOSTA E ACCETTAZIONE (opzione 3). Ai soli candidati il server
         manda una PROPOSTA (l'envelope della Fase 4, senza modello) con
         messaggi Flower veri (MessageType.QUERY). Ogni candidato rilegge il
         proprio stato e accetta se riesce a fare almeno E_lo senza intaccare
         la riserva, altrimenti rifiuta; in entrambi i casi risponde col
         proprio SoC, workload e stato di carica, che diventano lo stato
         noto. I rifiuti vengono sostituiti dai successivi in classifica,
         per al massimo `max_waves` ondate. Il modello va SOLO a chi ha
         accettato: un rifiuto non spreca il download.
      COSTO: ogni scambio e' una sessione radio LTE (~12.6 J, vedi
      communication_model). Chi accetta la condivide col download del
      modello che segue, chi rifiuta la paga intera. Ogni ondata ritarda il
      round di promozione + RTT (~0.33 s).
      [B] un rifiuto conta come contatto: azzera l'anzianita', cosi' il
      client non torna subito in cima alla classifica e non si spreca un
      messaggio a ogni round; l'anzianita' poi ricresce, quindi non resta
      escluso per sempre (niente starvation).
      [B] se in `max_waves` ondate nessuno accetta, il pool e' considerato
      esaurito e la run termina, come quando l'ILP e' infeasible.
      [B] quanto la stima sbaglia lo misura smart_info_err (|SoC stimato -
      SoC del fuel gauge| medio sul pool), quanto costa la negoziazione
      energy_ctrl_wh e smart_overhead_s.
      [B] la negoziazione avviene PRIMA di WorldState.begin_round, che fa
      avanzare il workload e decide le ricariche: resta uno scarto fra cio'
      che il client ha dichiarato e cio' che trova al training. E' il motivo
      del late binding, e il client puo' ancora ritirarsi (opt-out).

    FASE 3 -- SELEZIONE
          score = a*SoC + b*D + c*min(1, eta / stale_max)     a + b = 1
      SoC e' quello NOTO AL SERVER (la stima della Fase 2a), D la
      divergenza JS delle classi (come SAGE), eta i round dall'ultima
      selezione e c = `stale_weight`.
      [B] IL TERMINE DI ANZIANITA'. Con il solo a*SoC + b*D, D e' statica:
      chi ha divergenza bassa non viene scelto mai, e chi consuma poco resta
      carico e viene riscelto sempre (misurato: 4 client su 30 mai
      selezionati, uno scelto 228 volte su 250, Gini 0.62 contro 0.19 di
      FedAvg). L'anzianita' fa guadagnare punteggio a chi non partecipa da
      tanto, finche' non viene provato. E' la Fase 8 della proposta, ed e'
      quello che fanno Oort e F3AST.
      [B] con score non negativi e soli vincoli di cardinalita' l'ILP
      coincide con i top-k per score, come in SAGE.

    FASE 4 -- PROPOSTA (envelope), per ogni selezionato
      E in [E_lo, E_hi]   B in {batch ammessi}   + deadline, soglia SoC
      E_lo = epochs_min: sotto, l'update non porta segnale utile.
      E_hi = min(2*E_nom, vincolo di deadline, vincolo di energia), dove
        deadline: E * t_epoca(B, w_noto) <= T_round
        energia:  E * dSoC_epoca <= SoC_noto - soc_min - dSoC_upload
                  con dSoC stimato LINEARMENTE (P*t/capacita') e corretto da
                  un fattore k_i appreso dai consumi che il client riporta.
      B: il batch ottimo del device (compute_model.optimal_batch_size) e i
      due adiacenti nella lista delle taglie ammesse.
      [B] e' una stima volutamente LASCA: il server non conosce il workload
      attuale ne' la curva vera della batteria. Il compito di essere precisi
      e' del client.
      [B] T_round = deadline_mult * mediana dei tempi nominali (E_nom, B_nom,
      senza workload). Limita lo straggler, quindi anche l'idle che gli altri
      pagano aspettandolo (modello ESCS), e tiene i tau comparabili per
      FedNova. Senza deadline un client in carica riceveva le epoche massime e
      arrivava a tau 30 volte gli altri: vedi aggregate_train.

    FASE 5 -- DECISIONE DEL CLIENT: vedi tesiFL/client_policy.py.
      Il client rilegge SoC, workload e carica, stima la loss del modello
      globale sui propri dati e sceglie (E, B) dentro l'envelope, oppure si
      ritira (opt-out) se nemmeno il minimo rispetta la sua riserva.

    FASE 6 -- TRAINING con guardia: il client si ferma se finisce il budget
      energetico, anche a meta' epoca, e lo dichiara.

    FASE 7 -- RISPOSTA E AGGREGAZIONE
      Il client comunica (E, B) scelti, passi eseguiti, SoC iniziale e
      FINALE, loss prima e dopo. Il server:
        - verifica che (E, B) stiano nell'envelope: se no scarta l'update;
        - addebita al mondo fisico il lavoro DICHIARATO (Fedavg._work_done);
        - aggrega con FedNova, tau_i = passi dichiarati dal client.

    FASE 8 -- AGGIORNAMENTO DELLO STATO
      SoC noto <- SoC finale riportato; k_i <- media mobile del rapporto fra
      calo di SoC osservato e stima lineare; divergenza D_i dalla
      distribuzione delle classi.

    MODELLO DI BATTERIA (battery_mode), i tre bracci del confronto
      "soc"     (nm, etichetta sage_smart) il SoC che client e server vedono
                e' quello VERO del mondo (modello da datasheet), e il client
                prevede il consumo con lo stesso modello. E' la variante
                della tesi.
      "peuk"    (sage_smart_peuk) il device usa un fuel gauge di PEUKERT
                (WorldState.soc_peukert, battery_model_peukert) e prevede il
                consumo con la stessa formula.
      "energy"  (sage_smart_lin) il device usa un fuel gauge LINEARE
                (WorldState.soc_linear): SoC_lin <- SoC_lin - P/eta * dt /
                (V_nom * C_nom), e prevede il consumo con la stessa formula.
      Con "peuk" ed "energy" client e server ragionano su quel SoC, ma la
      batteria vera si scarica col modello da datasheet, quindi la riserva
      protetta puo' non esserlo davvero. E' lo stesso confronto di
      sage/sage_peuk/sage_soc ed escs_sd_lin/escs_sd_peuk/escs_sd; qui i
      gauge seguono anche idle e ricarica, quindi i bracci differiscono SOLO
      per il modello di batteria.
      La colonna smart_soc_err (|SoC dichiarato - SoC vero|) vale ~0 con
      "soc" e misura l'errore del modello con "peuk" ed "energy";
      soc_believed_mean e' il SoC medio creduto, da confrontare con mean_soc.

    COSA SA IL SERVER, E DA DOVE
      Le decisioni della strategia usano solo: profili statici (Fase 0),
      risposte alle proposte (Fase 2), metriche riportate dai client
      (Fase 7). Il mondo fisico entra in due soli punti, entrambi fuori
      dalle decisioni:
        - `_device_record`: i SENSORI del device, allegati al messaggio per
          conto del device stesso (in un telefono vero li legge dall'OS);
        - le colonne smart_soc_err e soc_believed_mean, che confrontano
          creduto e vero: sono verifiche, non input.

    PARAMETRI
      stale_weight, stale_max  peso (c) e scala dell'anzianita'
      soc_min        riserva che il CLIENT protegge (vincolo 2e di SAGE)
      epochs_min     E_lo dell'envelope; E_hi al massimo epochs_cap_mult*E
      batch_choices  taglie di batch ammesse
      deadline_mult  scala della deadline sulla mediana nominale
      probe_samples  campioni per stimare la loss prima di decidere
      max_waves      ondate di proposte per round (1 = nessun rimpiazzo)
      query_timeout_s  attesa massima delle risposte ai messaggi di controllo
      calib_alpha    peso della media mobile su k_i
      battery_mode   "soc" (datasheet) | "peuk" (Peukert) | "energy" (lineare)
    """

    def __init__(
        self,
        *args,
        stale_weight: float = 0.3,
        stale_max: int = 25,
        soc_min: float = 0.20,
        batch_choices: tuple[int, ...] = (32, 64, 128, 256),
        epochs_cap_mult: int = 2,
        epochs_min: int = 2,
        partition_sizes: dict[int, int] | None = None,
        deadline_mult: float = 1.0,
        probe_samples: int = 256,
        max_waves: int = 2,
        query_timeout_s: float = 600.0,
        calib_alpha: float = 0.3,
        battery_mode: str = "soc",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if battery_mode not in BATTERY_MODELS:
            raise ValueError(f"battery_mode: uno fra {tuple(BATTERY_MODELS)}")
        self.stale_w = float(stale_weight)
        self.stale_max = int(stale_max)
        self.soc_min = float(soc_min)
        self.batch_choices = tuple(int(b) for b in batch_choices)
        self.epochs_cap = int(epochs_cap_mult) * int(self.epochs)
        self.epochs_min = int(epochs_min)
        self.deadline_mult = float(deadline_mult)
        self.probe_samples = int(probe_samples)
        self.max_waves = max(1, int(max_waves))
        self.query_timeout_s = float(query_timeout_s)
        self.calib_alpha = float(calib_alpha)
        self.battery_mode = battery_mode
        # modello del fuel gauge del device: nm (vero), lin o peuk
        self.model = BATTERY_MODELS[battery_mode]
        # Fase 0: taglia dei dataset. Se manca, la si impara dalle risposte.
        self._n: dict[int, int] = {int(c): int(n) for c, n in
                                   (partition_sizes or {}).items()}
        # [B] workload ignoto = caso peggiore (proposta, Fase 1): meglio
        # proporre poco lavoro a un client che si scopre libero che troppo a
        # uno che si scopre occupato.
        self.w_busy = float(max(WORKLOAD_STATES.values()))

        self._last_seen: dict[int, int] = {}       # cid -> ultimo round scelto
        self._plan: dict[int, Envelope] = {}       # cid -> envelope del round
        self._batch_of: dict[int, int] = {}

        # stato che il server tiene per ogni client: l'ultimo REPORT (cosa il
        # client ha detto e quando) e lo stato NOTO (stima o report fresco)
        self._subscribed = False
        self._deadline_s: float = math.inf
        self._clock_s = 0.0                        # tempo simulato, inizio round
        self._soc_rep: dict[int, float] = {}
        self._t_rep: dict[int, float] = {}
        self._w_rep: dict[int, float] = {}
        self._chg_rep: dict[int, bool] = {}
        self._soc_known: dict[int, float] = {}
        self._chg_known: dict[int, bool] = {}
        self._w_known: dict[int, float] = {}
        self._k_cal: dict[int, float] = {}
        self._loss_known: dict[int, float] = {}
        # per round: vincolo che ha fissato E_hi, risposte dei client e costi
        # della Fase 2
        self._bound: dict[int, str] = {}
        self._reports: dict[int, dict] = {}
        self._grid = None
        self._queried: set[int] = set()
        self._overhead_s = 0.0
        self._waves = 0
        self._n_queried = 0
        self._n_refused = 0
        self._info_err = float("nan")

    @property
    def variant(self) -> str:
        return "sage-smart" + {"nm": "", "lin": "-lin", "peuk": "-peuk"}[self.model]

    def configure_train(self, server_round, arrays, config, grid):
        # la griglia serve alla Fase 2 per i messaggi di controllo, che
        # partono dentro _select_clients prima dei messaggi di training
        self._grid = grid
        return super().configure_train(server_round, arrays, config, grid)

    def _round_overhead_s(self) -> float:
        return self._overhead_s

    # ===================================================== Fase 0 e Fase 2
    def _sensor_soc(self, cid: int) -> float:
        """SoC che il device legge dal proprio fuel gauge."""
        if self.model == "lin":
            return float(self.world.soc_linear(cid))
        if self.model == "peuk":
            return float(self.world.soc_peukert(cid))
        return float(self.world.snapshot(cid).soc)

    def _believed_soc_mean(self) -> float | None:
        """SoC medio creduto dai device (fuel gauge), non la stima del server."""
        cids = list(self._node_to_cid.values())
        return float(np.mean([self._sensor_soc(c) for c in cids])) if cids else None

    def _subscribe(self) -> None:
        """Iscrizione: dati statici e SoC iniziale di ogni client."""
        cids = list(self._node_to_cid.values())
        for c in cids:
            # il primo "report" e' l'iscrizione, al tempo 0, workload ignoto
            self._record(c, self._sensor_soc(c), self.w_busy, False, 0.0)
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
            MODEL_NAMES[self.model])

    def _record(self, cid: int, soc: float, w: float, chg: bool, t: float) -> None:
        """Un report del client: diventa anche lo stato noto."""
        self._soc_rep[cid], self._w_rep[cid] = float(soc), float(w)
        self._chg_rep[cid], self._t_rep[cid] = bool(chg), float(t)
        self._soc_known[cid], self._w_known[cid] = float(soc), float(w)
        self._chg_known[cid] = bool(chg)

    def _estimate(self, cid: int) -> float:
        """Fase 2a: SoC estrapolato dall'ultimo report col consumo di idle.

            SoC_est = SoC_rep - P_idle(w_rep) * dt / (V_nom * C_nom)

        [B] lineare e senza k_i: k_i e' calibrato sul training (correnti alte),
        in idle la corrente e' bassa e il modello lineare sbaglia poco.
        In carica al momento del report: si assume invariato, perche' non si
        sa quando la carica finisce (stima prudente, la carica lo alzerebbe).
        """
        soc = self._soc_rep[cid]
        if self._chg_rep.get(cid, False):
            return soc
        dt = max(0.0, self._clock_s - self._t_rep.get(cid, 0.0))
        w = self._w_rep.get(cid, self.w_busy)
        p_idle = IDLE_POWER_W + w * (TRAIN_POWER_W - IDLE_POWER_W)
        cap_wh = self.world.profile(cid).battery_capacity_mah / 1000.0 * V_NOMINAL
        return float(np.clip(soc - energy_wh_linear(p_idle, dt) / cap_wh, 0.0, 1.0))

    def _refresh_state(self, available: list[int]) -> None:
        """Fase 2a: lo stato con cui il server ordina i client."""
        for c in self._node_to_cid.values():
            self._soc_known[c] = self._estimate(c)
            self._chg_known[c] = self._chg_rep[c]
            self._w_known[c] = self._w_rep[c]
        # [B] verifica, non input: quanto lo stato noto si discosta da quello
        # che il device leggerebbe adesso, sul pool selezionabile
        pool = [self._node_to_cid[n] for n in available]
        if pool:
            self._info_err = float(np.mean(
                [abs(self._soc_known[c] - self._sensor_soc(c)) for c in pool]))

    def _propose(self, nodes: list[int],
                 envs: dict[int, Envelope]) -> dict[int, Message]:
        """Fase 2b: proposte (MessageType.QUERY) e lettura delle risposte.

        Le risposte aggiornano subito lo stato noto. Il costo energetico NON
        si addebita qui: dipende da chi verra' selezionato (sessione radio
        condivisa col download del modello o no), vedi _charge_control.
        """
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
            self.world.apply_control(self._node_to_cid[n], merged=n in sel)

    def _resource(self, cid: int) -> float:
        """SoC usato nello score: quello NOTO al server, non quello vero."""
        return self._soc_known.get(cid, float(self.world.initial_soc(cid)))

    def _n_of(self, cid: int) -> int | None:
        if cid in self._n:
            return self._n[cid]
        return int(np.median(list(self._n.values()))) if self._n else None

    def _batch_for(self, cid: int) -> int:
        """Batch ottimo del device, calcolato una volta sola: dipende dal
        profilo hardware, che non cambia durante la run. [B] e' il minimo
        analitico del costo di un'epoca (compute_model.optimal_batch_size)."""
        if cid not in self._batch_of:
            b = optimal_batch_size(self.world.profile(cid))
            self._batch_of[cid] = min(self.batch_choices,
                                      key=lambda x: abs(x - b))
        return self._batch_of[cid]

    def _staleness(self, cid: int, server_round: int) -> float:
        """Round dall'ultima selezione, normalizzati su stale_max.

        Chi non e' mai stato scelto ha anzianita' massima dal primo round:
        e' il modo per provarlo invece di escluderlo per sempre.
        """
        last = self._last_seen.get(cid)
        age = server_round if last is None else server_round - last
        return float(min(1.0, age / max(1, self.stale_max)))

    # ============================================================ Fase 3
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
        if not k or not available:
            log(INFO, "%s round %s: pool vuoto", self.variant, server_round)
            return []
        self._refresh_state(available)
        cid_of = self._node_to_cid
        pool = available
        m = len(pool)
        hi = min(self.k + server_round - 1, int(np.floor(self.max_frac * m)))
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
        # la prima ondata di proposte e' esattamente la sua scelta, i
        # rimpiazzi sono i successivi nella stessa classifica
        ranking = sorted(pool, key=lambda n: -score[n])
        selected = self._negotiate(ranking, len(selected), server_round)
        if not selected:
            self._charge_control([])
            log(INFO, "%s round %s: nessun candidato ha accettato in %s "
                "ondate: pool esaurito", self.variant, server_round,
                self._waves)
            return []

        self._charge_control(selected)
        self._overhead_s = self._waves * ctrl_latency_s()
        for n in selected:
            self._last_seen[cid_of[n]] = server_round
        # envelope con lo stato appena riportato dai client che hanno accettato
        self._plan = {cid_of[n]: self._assign_work(cid_of[n]) for n in selected}

        E = [env.e_hi for env in self._plan.values()]
        never = sum(1 for c in cid_of.values() if c not in self._last_seen)
        log(INFO, "%s round %s: pool %s, %s selezionati (cap %s) | "
            "E_hi %s-%s (medio %.1f) | mai visti %s | interrogati %s, "
            "rifiuti %s, ondate %s",
            self.variant, server_round, m, len(selected), hi, min(E), max(E),
            float(np.mean(E)), never, self._n_queried, self._n_refused,
            self._waves)
        return selected

    def _negotiate(self, ranking: list[int], slots: int,
                   server_round: int) -> list[int]:
        """Fase 2b: proposte ai candidati, rifiuti rimpiazzati.

        Ondata 1: i primi `slots` della classifica. Ondata i: tanti nuovi
        candidati quanti rifiuti nell'ondata precedente, scorrendo la
        classifica. Al massimo `max_waves` ondate: oltre, il round parte con
        meno client (il costo di un'altra ondata e' una sessione radio per
        candidato e un altro ritardo sul round).
        """
        cid_of = self._node_to_cid
        accepted: list[int] = []
        asked: set[int] = set()
        while len(accepted) < slots and self._waves < self.max_waves:
            batch = [n for n in ranking if n not in asked][: slots - len(accepted)]
            if not batch:
                break
            self._waves += 1
            asked.update(batch)
            envs = {cid_of[n]: self._assign_work(cid_of[n]) for n in batch}
            replies = self._propose(batch, envs)
            for n in batch:
                r = replies.get(n)
                if r is not None and int(self._rep(r, "smart-accept", 0)) == 1:
                    accepted.append(n)
                else:
                    self._n_refused += 1
                    # [B] contattato: l'anzianita' riparte da zero, cosi' non
                    # torna in testa alla classifica al round successivo
                    self._last_seen[cid_of[n]] = server_round
        return accepted

    def _mark_half(self, selected: list[int]) -> None:
        """Disattivata: le mezze epoche di SAGE qui non servono, la riduzione
        del lavoro per batteria scarica la decide il client."""
        self._half = set()

    # ============================================================ Fase 4
    def _batches_for(self, cid: int) -> tuple[int, ...]:
        """Batch ottimo del device e i due adiacenti fra le taglie ammesse."""
        ch = self.batch_choices
        i = ch.index(self._batch_for(cid))
        return tuple(ch[max(0, i - 1): i + 2])

    def _assign_work(self, cid: int) -> Envelope:
        prof = self.world.profile(cid)          # statico: dalla Fase 0
        batches = self._batches_for(cid)
        b_ref = self._batch_for(cid)
        peff = effective_profile(prof, self._w_known.get(cid, self.w_busy))
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
                training_power_w(b_ref), training_time_s(peff, 1, n, b_ref)
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
        rec = DeviceReading.from_world(self.world, cid, model=self.model).to_record()
        rec["dataset-size"] = int(self._n_of(cid) or 0)
        return ConfigRecord(rec)

    # ============================================================ Fase 7
    def _rep(self, reply: Message, key: str, default=None):
        return self._metric(reply, key, default=default)

    def _is_optout(self, reply: Message) -> bool:
        return int(self._rep(reply, "smart-optout", 0) or 0) == 1

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
        """tau_i di FedNova: i passi che il client dichiara di aver fatto."""
        if self._is_optout(reply):
            return 0.0
        return float(self._rep(reply, "smart-steps", 0))

    # ============================================================ Fase 8
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
        # partono le stime dei round successivi (Fase 2a).
        self._record(cid, soc1, w, chg, self._clock_s + self._overhead_s + dt_s)
        if n_ex > 0:
            self._n[cid] = int(n_ex)
        loss = self._rep(reply, "smart-loss-pre", None)
        if loss is not None:
            self._loss_known[cid] = float(loss)

        # calibrazione k_i: calo di SoC dichiarato / stima lineare dello
        # stesso lavoro, con la durata ricostruita dal workload DICHIARATO.
        # [B] con battery_mode="energy" il client dichiara un SoC lineare,
        # quindi k_i resta 1; con "peuk" k_i impara il rapporto Peukert /
        # lineare. In nessuno dei due casi il server vede l'errore del gauge.
        if not chg and steps > 0 and n_ex > 0:
            prof = self.world.profile(cid)
            peff = effective_profile(prof, w)
            e_eff = steps / math.ceil(n_ex / b)
            t = training_time_s(peff, e_eff, n_ex, b) + probe_time_s(peff, probe)
            cap_wh = prof.battery_capacity_mah / 1000.0 * V_NOMINAL
            lin = (energy_wh_linear(training_power_w(b), t)
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
            # Fase 2: qualita' e costo dello stato usato per selezionare
            "smart_info_err": float(self._info_err),
            "smart_n_queried": int(self._n_queried),
            "smart_n_refused": int(self._n_refused),
            "smart_waves": int(self._waves),
            "smart_overhead_s": float(self._overhead_s),
            # [B] SoC medio CREDUTO dai device (fuel gauge), su tutta la
            # popolazione come mean_soc: con "soc" coincide con mean_soc, con
            # "peuk" ed "energy" la differenza e' l'errore del modello
            "soc_believed_mean": self._believed_soc_mean(),
        }
        log(INFO, "%s: %s risposte | E eseguite %.1f (min %.2f, max %.2f) "
            "| E_hi medio %.1f (energia %s, deadline %s) | opt-out %s | "
            "fuori deadline %s | errore SoC %.1e | errore stato noto %.3f",
            self.variant, len(rep), st["mean_epochs"], st["smart_min_epochs"],
            st["max_epochs"], st["smart_mean_e_hi"], st["smart_n_bound_energy"],
            st["smart_n_bound_deadline"], st["smart_n_optout"],
            st["smart_n_deadline_miss"], st["smart_soc_err"],
            st["smart_info_err"])
        return st

    # -------------------------------------------------- aggregazione FedNova
    def aggregate_train(self, server_round, replies):
        """Aggregazione step-normalized (FedNova, Wang et al. NeurIPS 2020).

            w_i     = p_i = |d_i| / |D|          peso dai soli dati
            tau_i   = passi di gradiente DICHIARATI dal client
            tau_eff = sum_i p_i * tau_i          passi efficaci
            x <- x + tau_eff * sum_i p_i * (Delta_i / tau_i)

        La divisione per tau_i e' la normalizzazione: sottopesa chi ha fatto
        piu' passi. tau_eff davanti evita che il passo globale si rimpicciolisca
        per effetto della divisione.

        [B] QUANDO tau E' UGUALE PER TUTTI QUESTA E' ESATTAMENTE FedAvg:
        tau_eff = tau e la formula si riduce alla media pesata per p_i.

        [B] tau DIPENDE ANCHE DA n_i, non solo da E e B. Con partizioni
        Dirichlet n varia di 4-5 volte, quindi la distorsione esiste anche a
        E e B fissi: FedAvg standard e' gia' sbilanciato nelle baseline, e i
        paper originali non lo correggono. Qui serve perche' (E, B) variano
        per client per costruzione.

        [B] CIO' CHE CONTA E' LA DISPERSIONE DI tau, NON IL SUO VALORE.
        L'update di ogni client viene moltiplicato per tau_eff/tau_i: il
        client con tau minimo e' amplificato di quel rapporto. Nella versione
        con (E, B) assegnati dal server, un client collegato alla rete
        riceveva le epoche massime a prescindere dai suoi dati e arrivava a
        tau=390 contro i 13 degli altri; l'update del client marginale pesava
        trenta volte quello del client in carica e l'accuratezza globale
        crollava da 0.50 a 0.14 finche' quel device restava in carica. Da qui
        la deadline nell'envelope e la colonna tau_spread, la grandezza da
        sorvegliare.

        [B] tau_i e' quello DICHIARATO dal client, non ricostruito da E e B:
        con l'early stopping (Fase 6) il client puo' fermarsi a meta' epoca.
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
        glob = [self._global_arrays[k].numpy().astype(np.float64) for k in keys]
        rows = []
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

        if len(rows) < 2:
            return arrays, metrics

        tot = sum(n for n, _, _ in rows)
        p = [n / tot for n, _, _ in rows]
        tau_eff = sum(pi * tau for pi, (_, tau, _) in zip(p, rows))

        out = [g.copy() for g in glob]
        for pi, (_, tau, w) in zip(p, rows):
            for j, wj in enumerate(w):
                out[j] += tau_eff * pi * (wj.astype(np.float64) - glob[j]) / tau

        taus = [tau for _, tau, _ in rows]
        self._round_stats.setdefault(int(server_round), {}).update({
            "tau_min": int(min(taus)),
            "tau_max": int(max(taus)),
            "tau_eff": float(tau_eff),
            # rapporto di amplificazione del client marginale: sopra ~6-8
            # l'aggregazione comincia a essere dominata da un solo update
            "tau_spread": float(max(taus) / max(1.0, min(taus))),
        })

        log(INFO, "round %s: FedNova su %s client | tau %s-%s, tau_eff %.1f, "
            "spread %.1f", server_round, len(rows), int(min(taus)),
            int(max(taus)), tau_eff, max(taus) / max(1.0, min(taus)))

        merged = ArrayRecord()
        for k, o in zip(keys, out):
            merged[k] = Array(o.astype(np.float32))
        return merged, metrics
