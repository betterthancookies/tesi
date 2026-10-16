from __future__ import annotations

from logging import INFO

import numpy as np
from flwr.app import (
    ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict,
)
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from device.communication_model import ctrl_latency_s
from device.world_state import WorldState


class PhysicalFedAvg(FedAvg):
    """FedAvg sul mondo fisico, base di tutte le strategie.

    BROADCAST DELLO STATO (opzione 1 del documento "Client Server Protocol"
    del relatore, call del 16/10). Se `broadcast` e' attivo, a inizio round
    il server chiede lo stato a tutti i client vivi (o ai primi
    `broadcast_top_m`, nota 3 del documento) e:
      - una frazione `broadcast_nonresp` non risponde (fuori copertura,
        spento): e' esclusa dalla selezione di QUESTO round;
      - chi risponde paga uno scambio di controllo: sessione radio intera se
        poi non viene selezionato, il solo payload se viene selezionato
        (subito dopo riceve il modello, stessa sessione);
      - il round si allunga della latenza dello scambio, o del timeout
        `broadcast_timeout_s` se qualcuno non risponde.
    [B] NEL SIMULATORE IL BROADCAST NON CAMBIA L'INFORMAZIONE: i client non
    hanno una vita autonoma da cui leggere un valore diverso da quello che il
    mondo gia' conosce. SAGE ed ESCS usano il SoC del mondo, SAGE-smart la sua
    stima (errore misurato < 1 punto di SoC): fanno le veci della risposta al
    broadcast. Il broadcast c'e' per CONTARNE IL COSTO: se si dichiara di
    avere un SoC fidato a inizio round, l'energia e il tempo per ottenerlo
    vanno pagati (richiesta del relatore, con ~10% di client che non
    rispondono). Si attiva per le strategie che selezionano sullo stato
    (SAGE, ESCS, SAGE-smart), non per FedAvg e FedProx che scelgono a caso.
    """

    def __init__(
        self,
        world: WorldState,
        epochs: int = 5,
        batch_size: int = 128,
        k_per_round: int = 10,
        selection_seed: int = 7,
        broadcast: bool = False,
        broadcast_nonresp: float = 0.10,
        broadcast_timeout_s: float = 1.0,
        broadcast_top_m: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.world = world
        self.epochs = epochs
        self.batch_size = batch_size
        self.k_per_round = k_per_round
        self._rng = np.random.default_rng(selection_seed)
        self.broadcast = bool(broadcast)
        self.bcast_nonresp = float(broadcast_nonresp)
        self.bcast_timeout_s = float(broadcast_timeout_s)
        self.bcast_top_m = int(broadcast_top_m)
        # [B] RNG propria: i non rispondenti non devono spostare la sequenza
        # della selezione casuale ne' quella di altri componenti
        self._bcast_rng = np.random.default_rng(selection_seed + 104729)
        self._bcast_responders: list[int] = []
        self._bcast_round: dict = {}
        self._bcast_overhead_s = 0.0
        self._bcast_energy_wh = 0.0
        # client VIVI del round, prima di escludere chi non risponde al
        # broadcast: e' il pool su cui SAGE e SAGE-smart calcolano il tetto 0.2
        self._n_alive = 0
        self._node_to_cid: dict[int, int] = {}
        self._epochs_sent: dict[int, int] = {}   # cid -> epoche del round in corso
        # [B] il batch ora e' per client: in SAGE-smart lo sceglie il client,
        # e l'addebito energetico deve usare quello EFFETTIVO (la potenza di
        # training scala con B) e non il valore globale.
        self._batch_sent: dict[int, int] = {}
        # [B] i pesi globali inviati nel round: servono a FedNova per
        # ricavare Delta_i = w_i - w_globale. FedAvg standard non ne ha
        # bisogno perche' media direttamente i pesi finali.
        self._global_arrays = None
        self._round_stats: dict[int, dict] = {}  # per il CSV per-round
        # durata dell'ultimo round (training piu' lento + ritardo iniziale):
        # il server la conosce, ha aspettato le risposte
        self._last_round_s = 0.0
        # [B] selezione vuota = fine della run. Serve a sage_soc, che termina
        # per esaurimento del pool e non per numero di round: senza questo
        # flag Flower continuerebbe a valutare tutti i client fino a R_max.
        self._exhausted = False

    def _ensure_mapping(self, grid: Grid) -> None:
        if not self._node_to_cid:
            self._node_to_cid = {
                nid: cid for cid, nid in enumerate(sorted(grid.get_node_ids()))
            }

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        if self._exhausted:
            return []
        self._ensure_mapping(grid)
        self.world.tick_all()
        available = [
            nid for nid, cid in self._node_to_cid.items()
            if self._is_available(cid)
        ]
        self._n_alive = len(available)
        if self.broadcast:
            available = self._broadcast(available, server_round)
        k = min(self.k_per_round, len(available))
        selected = self._select_clients(available, k, server_round)
        if not selected:
            self._exhausted = True
            log(INFO, "round %s: selezione vuota su %s disponibili, run terminata",
                server_round, len(available))
            return []
        if self.broadcast:
            self._charge_broadcast(selected)

        # [B] il workload avanza qui, prima che il server calcoli le durate e
        # selezioni: cosi' la selezione di questo round vede lo stato corrente,
        # non quello del round precedente.
        self.world.begin_round()

        self._global_arrays = arrays

        base = dict(config)
        base["server-round"] = server_round
        base["epochs"] = self.epochs
        base["batch-size"] = self.batch_size

        self._epochs_sent.clear()
        self._batch_sent.clear()
        messages: list[Message] = []
        for nid in selected:
            cid = int(self._node_to_cid[int(nid)])
            cfg = ConfigRecord(dict(base))
            cfg["cid"] = cid
            self._per_client_config(cid, cfg, server_round)
            self._epochs_sent[cid] = int(cfg["epochs"])
            self._batch_sent[cid] = int(cfg["batch-size"])
            rec = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: cfg}
            )
            dev = self._device_record(cid)
            if dev is not None:
                rec["device"] = dev
            messages.extend(
                self._construct_messages(rec, [int(nid)], MessageType.TRAIN)
            )

        log(INFO, "round %s: %s/%s disponibili, %s selezionati",
            server_round, len(available), len(self._node_to_cid), len(selected))
        return messages

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        if self._exhausted:
            return []
        self._ensure_mapping(grid)
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        for msg in messages:
            cid = self._node_to_cid.get(int(msg.metadata.dst_node_id))
            if cid is None:
                continue
            if self.configrecord_key not in msg.content:
                msg.content[self.configrecord_key] = ConfigRecord()
            msg.content[self.configrecord_key]["cid"] = int(cid)
        return messages

    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []
        idx = self._rng.choice(len(available), size=k, replace=False)
        return [available[i] for i in idx]

    # ------------------------------------------------------------ broadcast
    def _broadcast_targets(self, available: list[int], server_round: int) -> list[int]:
        """A chi chiedere lo stato. Default: a tutti i vivi. SAGE-smart, che
        ha una stima con cui ordinare, puo' limitarsi ai primi M (nota 3)."""
        return list(available)

    def _broadcast(self, available: list[int], server_round: int) -> list[int]:
        """Richiesta di stato a inizio round: ritorna chi ha risposto.

        Chi non risponde e' escluso dalla selezione di questo round. La
        latenza e' quella di uno scambio di controllo (promozione + RTT +
        payload); se qualcuno non risponde il server aspetta il timeout.
        """
        targets = self._broadcast_targets(available, server_round)
        resp = [n for n in targets if self._bcast_rng.random() >= self.bcast_nonresp]
        self._bcast_responders = resp
        lat = ctrl_latency_s()
        self._bcast_overhead_s = (max(lat, self.bcast_timeout_s)
                                  if len(resp) < len(targets) else lat)
        self._bcast_round = {"bcast_targets": len(targets), "bcast_resp": len(resp)}
        return resp

    def _charge_broadcast(self, selected: list[int]) -> None:
        """Addebita gli scambi del broadcast. Il selezionato riceve subito
        dopo il modello: lo scambio cade nella stessa sessione radio e paga
        il solo payload. Gli altri pagano una sessione intera."""
        sel = {int(n) for n in selected}
        for n in self._bcast_responders:
            e, _ = self.world.apply_control(self._node_to_cid[int(n)],
                                            merged=int(n) in sel)
            self._bcast_energy_wh += e

    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        return None

    def _device_record(self, cid: int) -> ConfigRecord | None:
        """Sensori del device simulato da allegare al messaggio di training.

        None per tutte le strategie server-side: il client non ne ha bisogno.
        sage_smart lo usa per dare al client cio' che un telefono vero
        leggerebbe dal sistema operativo (SoC, workload, carica).
        """
        return None

    def _work_done(self, cid: int, n_ex: int, reply: Message) -> tuple[float, int]:
        """(durata in s, batch) del lavoro che il client ha DAVVERO eseguito.

        Default: quello assegnato dal server, che il client esegue per intero.
        sage_smart lo sostituisce con quello che il client dichiara, perche'
        li' (E, B) li decide il client.
        """
        b = self._batch_sent.get(cid, self.batch_size)
        dt = self.world.round_duration_s(
            cid,
            epochs=self._epochs_sent.get(cid, self.epochs),
            batch_size=b,
            dataset_size_local=n_ex,
        )
        return dt, b

    def _round_overhead_s(self) -> float:
        """Secondi spesi prima del training (scambi di controllo): il
        broadcast dello stato, se attivo. sage_smart aggiunge la negoziazione."""
        return self._bcast_overhead_s if self.broadcast else 0.0

    def _aggregatable(self, replies: list[Message]) -> list[Message]:
        """Reply che entrano nella media. Default: tutte.

        sage_smart esclude i client che si sono ritirati dal round (opt-out):
        hanno pagato la comunicazione ma non hanno un update da mediare.
        """
        return replies

    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        """Hook per le sottoclassi, chiamato dopo l'addebito fisico.

        e_wh      Wh di carica pagati dalla batteria del mondo (training +
                  upload): e_wh / (V_nom * C_nom) e' il calo di SoC esatto
        e_lin_wh  Wh al carico, P*dt
        """
        return None

    def _is_available(self, cid: int) -> bool:
        return self.world.snapshot(cid).available

    def _metric(self, reply: Message, *keys, default=None):
        for name in sorted(reply.content.metric_records):
            rec = reply.content.metric_records[name]
            for key in keys:
                if key in rec:
                    return rec[key]
        return default

    def aggregate_train(
        self, server_round: int, replies: list[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        # nessuna reply: niente da aggregare, e super() dividerebbe per zero
        self._last_round_s = float(self._round_overhead_s())
        if not replies:
            return None, None

        replies = sorted(
            replies,
            key=lambda r: self._node_to_cid.get(int(r.metadata.src_node_id), 1 << 62),
        )

        durations: dict[int, float] = {}
        for reply in replies:
            if not reply.has_content():
                continue
            cid = self._node_to_cid[reply.metadata.src_node_id]
            n_ex = int(self._metric(reply, "num-examples", "num_examples"))
            dt, b = self._work_done(cid, n_ex, reply)
            durations[cid] = dt
            e_tr, l_tr = self.world.apply_round(cid, dt_s=dt, batch_size=b)
            e_cm, l_cm = self.world.apply_communication(cid)
            self._on_round_result(cid, n_ex, dt, float(e_tr + e_cm),
                                  float(l_tr + l_cm), reply, server_round)

        # [B] idle DOPO il training: il round dura quanto il selezionato piu'
        # lento, e ogni client paga la differenza (modello ESCS Sez. 4.2). Chi
        # non e' stato selezionato paga l'intera durata. L'ordine conta: se
        # l'idle venisse prima, un client al limite morirebbe di idle invece
        # che di training e il conteggio cambierebbe.
        # [B] il ritardo prima del training (messaggi di controllo di
        # sage_smart) allunga il round: in idle lo pagano tutti tranne i
        # selezionati, che in quel tempo stanno parlando col server.
        overhead = float(self._round_overhead_s())
        self._last_round_s = max(durations.values(), default=0.0) + overhead
        self.world.apply_idle({c: d + overhead for c, d in durations.items()})

        s = self.world.stats()
        self._round_stats[int(server_round)] = dict(s)
        if self.broadcast:
            # [B] l'energia del broadcast e' gia' dentro energy_ctrl_wh (e
            # quindi nel totale): qui la si riporta a parte per leggerla
            self._round_stats[int(server_round)].update({
                **self._bcast_round,
                "bcast_overhead_s": float(self._bcast_overhead_s),
                "energy_bcast_wh": float(self._bcast_energy_wh),
            })
        log(INFO, "round %s: %.3f Wh (train %.3f + comm %.3f) | SoC medio %.2f "
            "| in ricarica %s | falliti %s",
            server_round, s["total_energy_wh"], s["energy_train_wh"],
            s["energy_comm_wh"], s["mean_soc"], s["n_recharging"], s["n_failed"])

        replies = self._aggregatable(replies)
        if not replies:
            return None, None
        return super().aggregate_train(server_round, replies)