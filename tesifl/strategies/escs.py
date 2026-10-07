from __future__ import annotations

import math
from logging import INFO
from typing import Iterable

import numpy as np
from flwr.app import Message, MetricRecord
from flwr.common import log

from device.battery_model import V_NOMINAL
from strategies.fedavg import PhysicalFedAvg


class PhysicalESCS(PhysicalFedAvg):
    """ESCS (Maciel et al., 2024), allineato al paper e ai sorgenti degli autori.

    Round 1:     tutti i client disponibili (Alg. 1 righe 5-6; escs_sd.py
                 riga 108). Disattivabile con first_round_all=False.
    Filtro R:    (batteria - consumo previsto) >= min_battery  and  nq >= nq_min
    Utilita' U:
      "s" sistemica  U = (a*bat + b*(1 - cpu)*th + g*nq)^(1 - loss/max_loss)
      "m" modello    U = (loss/max_loss)^(1 - base)
    Penalita' th:  exp(|cpu - lim|) - 1  se cpu > lim,  1 altrimenti
    Selezione:     "d" i top-k per utilita', "p" i client con rand() <= U.

    da cui escs-sd, escs-sp, escs-md, escs-mp.

    LATENZA (cpu). Nel paper cpu = total_train_latency / max_training_latency
    e' un rapporto fra dati di PROFILO, noti per ogni client prima di partire,
    quindi cpu in (0, 1]. Qui la latenza di profilo e' predetta dal modello di
    calcolo del world state sulla VERA taglia della partizione locale
    (`partition_sizes`, ricostruita dal server con lo stesso partitioner dei
    client). Con device omogenei e' identica a quella misurata a fine round,
    quindi non serve aggiornarla. lim = percentile 95 / max, calcolato una
    volta sola su tutti i client (escs_sd.py righe 86-93; Table 4 dice "95%
    of maximum", il codice usa il percentile).

    [B] una latenza di riferimento su un dataset nominale diverso da quello
    vero manda cpu sopra 1, il termine (1-cpu)*th diventa molto negativo e
    l'utilita' sistemica collassa a zero: batteria e rete smettono di contare.
    E' il motivo per cui il riferimento DEVE essere calcolato sulle taglie
    reali.

    LOSS. Il paper usa la loss di VALUTAZIONE (c_l, Table 2) e con eval
    fraction 1.0 la conosce per tutti i client a ogni round (Alg. 1 righe
    13-14). Qui si legge da aggregate_evaluate: dopo il round 1 ogni client
    ha una loss. Chi non ne ha ancora (fraction_evaluate < 1) vale
    loss = max_loss, cioe' U = 1: priorita' all'esplorazione, che e' l'intento
    dell'inf iniziale nel codice originale (li' pero' inf/inf = nan
    escludeva per sempre i mai osservati in -sp; qui e' corretto).

    RISORSA DI BATTERIA:

      battery_mode="lin"     (lin, etichetta escs_sd_lin) bat = SoC del fuel
                             gauge LINEARE (WorldState.soc_linear):
                             P/eta * dt / (V_nom * C_nom), idle e ricarica
                             inclusi.
      battery_mode="peuk"    (peuk, escs_sd_peuk) bat = SoC del fuel gauge di
                             Peukert (WorldState.soc_peukert), idle e ricarica
                             inclusi.
      battery_mode="soc"     (nm, escs_sd) bat = SoC VERO, modello datasheet.
      battery_mode="energy"  (escs_sd_paper) bat = SoC_0 - Wh_LINEARI dei soli
                             round in cui il client lavora / capacita'. E' la
                             contabilita' server-side scritta quando il mondo
                             non aveva idle ne' ricarica: non vede l'idle
                             (circa meta' dell'energia totale) ne' le
                             ricariche. Resta per misurare quanto costa
                             ignorare l'idle, NON come modello lineare.

    lin, peuk e soc differiscono SOLO per il modello di batteria: e' il
    confronto puro, lo stesso di sage_lin/sage_peuk/sage_soc e di
    sage_smart_lin/_peuk/sage_smart. Se l'effetto si ripete su tre algoritmi
    diversi non dipende dall'algoritmo.
    La risorsa entra sia nell'utilita' sia nel filtro R, perche' nel paper
    sono la stessa grandezza; anche il consumo previsto e' nella stessa
    contabilita' della risorsa.

    [B] la modalita' probabilistica non ha tetto: seleziona un numero
    variabile di client, ed e' il motivo per cui nel paper -sp/-mp sono le
    varianti piu' energivore. Il ramo deterministico e' un fallback che
    scatta solo se la probabilistica seleziona zero client. Con
    cap_probabilistic=True si tronca a k per il confronto a parita' di
    client per round, ma -sp diventa quasi indistinguibile da -sd.

    ADATTAMENTI DICHIARATI
    - il consumo previsto e' quello dell'ultimo round del client, in frazione
      di capacita'. Nell'originale e' delta_train_battery, costante per
      client; qui varia col carico locale, quindi l'ultimo osservato e' la
      stima. Chi non e' mai stato osservato usa la media degli altri.
    - min_battery e nq_min sono globali. Nell'originale sono campi per client
      del profilo device, che `device/` non ha. [ASSUNZIONE]
    - nq e' estratta U(0,1) una volta per client, come trans_prob nel profilo
      degli autori. `device/` modella il costo della comunicazione, non la
      probabilita' di fallimento, quindi non c'e' un corrispettivo fisico.
    - pesi a = b = g = 1/3: il paper usa pesi uguali senza specificarne la
      somma. [ASSUNZIONE] la somma a 1 e' nostra.
    - il numero di selezionati e' k per entrambe le varianti dal round 2 in
      avanti. Nell'originale -sd usa round(num_clients * C) e -sp usa
      round(len(idonei) * C).
    """

    def __init__(
        self,
        *args,
        utility_mode: str = "s",
        selection_mode: str = "d",
        battery_mode: str = "soc",
        partition_sizes: dict[int, int] | None = None,
        first_round_all: bool = True,
        battery_weight: float = 1 / 3,
        cpu_cost_weight: float = 1 / 3,
        link_prob_weight: float = 1 / 3,
        time_percentile: float = 95.0,
        min_battery: float = 0.20,
        min_network_quality: float = 0.20,
        cap_probabilistic: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        um, sm = utility_mode.lower(), selection_mode.lower()
        if um not in ("s", "m") or sm not in ("d", "p"):
            raise ValueError("utility_mode in {s,m}, selection_mode in {d,p}")
        if battery_mode not in ("soc", "lin", "peuk", "energy"):
            raise ValueError("battery_mode: 'soc' (nm), 'lin', 'peuk' oppure 'energy' (paper)")
        if not partition_sizes:
            raise ValueError(
                "partition_sizes obbligatorio: senza le taglie reali la "
                "latenza di riferimento e' sbagliata e l'utilita' collassa"
            )
        self.utility_mode, self.selection_mode = um, sm
        self.battery_mode = battery_mode
        self.first_round_all = bool(first_round_all)
        self.w_b, self.w_c, self.w_l = (
            float(battery_weight), float(cpu_cost_weight), float(link_prob_weight)
        )
        self.time_percentile = float(time_percentile)
        self.min_battery = float(min_battery)
        self.min_nq = float(min_network_quality)
        self.cap_prob = bool(cap_probabilistic)
        self._sizes = {int(c): int(n) for c, n in partition_sizes.items()}

        self._loss: dict[int, float] = {}     # cid -> eval loss ultimo round
        self._cons: dict[int, float] = {}     # cid -> consumo ultimo round (frazione)
        self._used_wh: dict[int, float] = {}  # cid -> Wh LINEARI cumulati (mode energy)
        self._nq: dict[int, float] = {}       # cid -> qualita' rete, statica
        self._lat_ref: dict[int, float] = {}  # cid -> latenza di profilo
        self._max_lat = 0.0
        self._lim = 1.0
        self._cap_wh: dict[int, float] = {}

    @property
    def variant(self) -> str:
        suffix = {"soc": "", "lin": "-lin", "peuk": "-peuk",
                  "energy": "-paper"}[self.battery_mode]
        return f"escs-{self.utility_mode}{self.selection_mode}{suffix}"

    def _believed_soc_mean(self) -> float | None:
        """SoC medio che il selettore crede, per la colonna soc_believed_mean."""
        cids = list(self._node_to_cid.values())
        return float(np.mean([self._battery(c) for c in cids])) if cids else None

    # ----------------------------------------------------------- componenti
    def _ensure_profile(self) -> None:
        """Latenze di profilo e nq: una volta sola, su tutti i client."""
        if self._lat_ref:
            return
        for cid in sorted(self._node_to_cid.values()):
            if cid not in self._sizes:
                raise KeyError(f"partition_sizes non contiene il cid {cid}")
            # [B] durata NOMINALE, senza workload: nel paper la latenza di
            # training e' un dato di profilo statico, noto al server. Il
            # workload e' proprio cio' che il server non puo' conoscere in
            # anticipo, ed e' il motivo per cui la decisione andrebbe presa
            # dal client. Con i tier questa latenza ora varia per client, che
            # e' la situazione che il paper descrive.
            self._lat_ref[cid] = float(self.world.round_duration_s(
                cid, epochs=self.epochs, batch_size=self.batch_size,
                dataset_size_local=self._sizes[cid], include_workload=False,
            ))
            self._nq[cid] = float(self._rng.uniform(0.0, 1.0))
        lats = list(self._lat_ref.values())
        self._max_lat = max(lats) or 1.0
        self._lim = float(np.percentile(lats, self.time_percentile)) / self._max_lat
        log(INFO, "%s: latenze di profilo %.1f-%.1f s, lim=%.3f",
            self.variant, min(lats), self._max_lat, self._lim)

    def _capacity(self, cid: int) -> float:
        if cid not in self._cap_wh:
            mah = float(self.world.profile(cid).battery_capacity_mah)
            self._cap_wh[cid] = mah / 1000.0 * V_NOMINAL
        return self._cap_wh[cid]

    def _battery(self, cid: int) -> float:
        """La risorsa di batteria vista dal selettore."""
        if self.battery_mode == "soc":
            return float(self.world.snapshot(cid).soc)
        if self.battery_mode == "peuk":
            return float(self.world.soc_peukert(cid))
        if self.battery_mode == "lin":
            return float(self.world.soc_linear(cid))
        # "energy": contabilita' dei soli round (vedi docstring)
        cap = self._capacity(cid)
        if cap <= 0.0:
            return 1.0
        soc0 = float(self.world.initial_soc(cid))
        return float(np.clip(soc0 - self._used_wh.get(cid, 0.0) / cap, 0.0, 1.0))

    def _consumption(self, cid: int) -> float:
        if cid in self._cons:
            return self._cons[cid]
        return float(np.mean(list(self._cons.values()))) if self._cons else 0.0

    def _has_resources(self, cid: int) -> bool:
        return (self._battery(cid) - self._consumption(cid) >= self.min_battery
                and self._nq[cid] >= self.min_nq)

    # ------------------------------------------------------------ selezione
    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []
        self._ensure_profile()
        cid_of = self._node_to_cid

        # Alg. 1 righe 5-6: al primo round partecipano tutti
        if server_round == 1 and self.first_round_all:
            log(INFO, "%s round 1: tutti i %s disponibili", self.variant,
                len(available))
            return list(available)

        ast = [n for n in available if self._has_resources(cid_of[n])]

        max_loss = max(self._loss.values()) if self._loss else 1.0
        util = {n: self._utility(cid_of[n], max_loss) for n in ast}
        # tie-break randomizzato: senza, il sort stabile esplora nell'ordine
        # di `available`, accoppiato alla partizione dei dati.
        ast = [ast[i] for i in self._rng.permutation(len(ast))] if ast else []

        rate = float("nan")
        selected: list[int] = []
        if ast and self.selection_mode == "p":
            selected = [n for n in ast if self._rng.random() <= util[n]]
            rate = len(selected) / len(ast)
            if self.cap_prob and len(selected) > k:
                idx = self._rng.choice(len(selected), size=k, replace=False)
                selected = [selected[i] for i in idx]

        if not selected and ast:  # deterministica: default, e fallback della -p
            selected = (ast if len(ast) <= k
                        else sorted(ast, key=lambda n: -util[n])[:k])

        if not selected:
            # terzo fallback degli autori: si ignora la qualita' del link e si
            # seleziona per sola batteria, ordinando per loss decrescente.
            batt = [n for n in available
                    if self._battery(cid_of[n]) - self._consumption(cid_of[n])
                    >= self.min_battery]
            batt.sort(key=lambda n: -self._loss.get(cid_of[n], max_loss))
            selected = batt[:k]

        log(INFO, "%s round %s: %s idonei su %s, %s selezionati (accept %.2f)",
            self.variant, server_round, len(ast), len(available),
            len(selected), rate)
        return selected

    def _utility(self, cid: int, max_loss: float) -> float:
        bat = self._battery(cid)
        cpu = self._lat_ref[cid] / self._max_lat          # in (0, 1]
        th = math.exp(abs(cpu - self._lim)) - 1 if cpu > self._lim else 1.0
        base = max(
            self.w_b * bat + self.w_c * (1.0 - cpu) * th + self.w_l * self._nq[cid],
            1e-12,
        )
        ratio = min(max(self._loss.get(cid, max_loss) / max_loss, 1e-12), 1.0)
        try:
            return float(base ** (1.0 - ratio) if self.utility_mode == "s"
                         else ratio ** (1.0 - base))
        except (OverflowError, ValueError):
            return 0.0

    # ---------------------------------------------------------------- update
    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        # consumo previsto nella STESSA contabilita' della risorsa
        # (lineare per "energy" e "lin": il calo del gauge lineare per training
        # + upload e' esattamente e_lin_wh / capacita')
        spent = e_lin_wh if self.battery_mode in ("energy", "lin") else e_wh
        self._used_wh[cid] = self._used_wh.get(cid, 0.0) + e_lin_wh
        cap = self._capacity(cid)
        if self.battery_mode == "peuk":
            # calo del gauge di Peukert per training + upload di questo round
            self._cons[cid] = float(self._peuk_drop.get(cid, 0.0))
        elif cap > 0.0:
            self._cons[cid] = float(spent / cap)

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """Registra la loss di valutazione per client (c_l del paper)."""
        replies = list(replies)
        for reply in replies:
            if not reply.has_content():
                continue
            cid = self._node_to_cid.get(int(reply.metadata.src_node_id))
            if cid is None:
                continue
            loss = self._metric(reply, "eval_loss", "eval-loss", "loss", default=None)
            if loss is not None and math.isfinite(float(loss)):
                self._loss[cid] = float(loss)
        return super().aggregate_evaluate(server_round, replies)