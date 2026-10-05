from __future__ import annotations

from logging import INFO

import numpy as np
from flwr.app import (
    ArrayRecord, ConfigRecord, Message, MessageType, MetricRecord, RecordDict,
)
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from device.world_state import WorldState


class PhysicalFedAvg(FedAvg):
    def __init__(
        self,
        world: WorldState,
        epochs: int = 5,
        batch_size: int = 128,
        k_per_round: int = 10,
        selection_seed: int = 7,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.world = world
        self.epochs = epochs
        self.batch_size = batch_size
        self.k_per_round = k_per_round
        self._rng = np.random.default_rng(selection_seed)
        self._node_to_cid: dict[int, int] = {}
        self._epochs_sent: dict[int, int] = {}   # cid -> epoche del round in corso
        # [B] il batch ora e' per client: SAGE-smart lo assegna in modo
        # dinamico, e l'addebito energetico deve usare quello EFFETTIVO
        # (la potenza di training scala con B) e non il valore globale.
        self._batch_sent: dict[int, int] = {}
        # [B] i pesi globali inviati nel round: servono a FedNova per
        # ricavare Delta_i = w_i - w_globale. FedAvg standard non ne ha
        # bisogno perche' media direttamente i pesi finali.
        self._global_arrays = None
        self._round_stats: dict[int, dict] = {}  # per il CSV per-round
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
        k = min(self.k_per_round, len(available))
        selected = self._select_clients(available, k, server_round)
        if not selected:
            self._exhausted = True
            log(INFO, "round %s: selezione vuota su %s disponibili, run terminata",
                server_round, len(available))
            return []

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

    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        return None

    def _device_record(self, cid: int) -> ConfigRecord | None:
        """Sensori del device simulato da allegare al messaggio di training.

        None per tutte le strategie server-side: il client non ne ha bisogno.
        sage_smart2 lo usa per dare al client cio' che un telefono vero
        leggerebbe dal sistema operativo (SoC, workload, carica).
        """
        return None

    def _work_done(self, cid: int, n_ex: int, reply: Message) -> tuple[float, int]:
        """(durata in s, batch) del lavoro che il client ha DAVVERO eseguito.

        Default: quello assegnato dal server, che il client esegue per intero.
        sage_smart2 lo sostituisce con quello che il client dichiara, perche'
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

    def _aggregatable(self, replies: list[Message]) -> list[Message]:
        """Reply che entrano nella media. Default: tutte.

        sage_smart2 esclude i client che si sono ritirati dal round (opt-out):
        hanno pagato la comunicazione ma non hanno un update da mediare.
        """
        return replies

    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        """Hook per le sottoclassi, chiamato dopo l'addebito fisico.

        e_wh      Wh Peukert (carica realmente pagata dalla batteria)
        e_lin_wh  Wh lineari P*dt (quelli che vedono SAGE/ESCS originali)
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
        self.world.apply_idle(durations)

        s = self.world.stats()
        self._round_stats[int(server_round)] = dict(s)
        log(INFO, "round %s: %.3f Wh (train %.3f + comm %.3f) | SoC medio %.2f "
            "| in ricarica %s | falliti %s",
            server_round, s["total_energy_wh"], s["energy_train_wh"],
            s["energy_comm_wh"], s["mean_soc"], s["n_recharging"], s["n_failed"])

        replies = self._aggregatable(replies)
        if not replies:
            return None, None
        return super().aggregate_train(server_round, replies)