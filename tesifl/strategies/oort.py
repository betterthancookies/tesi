from __future__ import annotations

import math
from logging import INFO

import numpy as np
from flwr.app import Message
from flwr.common import log

from strategies.fedavg import PhysicalFedAvg


class PhysicalOort(PhysicalFedAvg):
    """Oort (Lai et al., OSDI 2021) sul mondo fisico di PhysicalFedAvg.

    Cambia SOLO la politica di selezione; bookkeeping energetico, training
    locale e aggregazione restano quelli di FedAvg, quindi il confronto con le
    altre baseline e' a parita' di ogni altro fattore.

    Utilita' di un client i, come da paper:

      statistical    U(i)  = |B_i| * sqrt( (1/|B_i|) * sum_k Loss(k)^2 )   (Eq. 1)
      staleness      U'(i) = U(i) + sqrt( 0.1 * ln(R) / R_i )              (Sez. 4.1)
      system         Util(i) = U'(i) * (T / t_i)^alpha   se t_i > T        (Eq. 2)
                             = U'(i)                     altrimenti

    dove R e' il round corrente, R_i l'ultimo round in cui i ha partecipato,
    t_i la durata del suo ultimo round e T la durata di riferimento (percentile
    delle durate osservate, adattato dal pacer).

    Selezione di k client:
      - (1-eps)*k per sfruttamento: si scartano i client sotto la soglia di
        cut-off (frazione dell'utilita' del miglior candidato) e si campiona
        senza rimpiazzo con probabilita' proporzionale all'utilita';
      - eps*k per esplorazione: campionamento uniforme tra i client mai
        selezionati; se non bastano, si completa tra i disponibili.
      - eps decade a ogni round (exploration_decay) fino a exploration_min.

    Pacer (Sez. 4.2): ogni pacer_step round si confronta l'utilita' media
    ottenuta nell'ultima finestra con quella precedente; se e' calata, il
    vincolo temporale T viene rilassato di pacer_delta percentili, perche'
    significa che si stanno escludendo client statisticamente utili solo
    perche' lenti.

    SCOSTAMENTI DICHIARATI rispetto al paper:
      - la blacklist dei client sovra-selezionati non e' implementata (nel
        paper serve contro l'over-fitting su popolazioni molto grandi; qui i
        client sono 100 e i round poche decine);
      - le loss per campione provengono dall'ultima epoca locale, non da una
        passata dedicata sul dataset completo.
    """

    def __init__(
        self,
        *args,
        exploration_factor: float = 0.9,   # eps iniziale
        exploration_decay: float = 0.98,
        exploration_min: float = 0.3,
        exploration_alpha: float = 0.3,    # alpha della system utility
        round_threshold: float = 30.0,     # percentile per T
        cut_off_util: float = 0.95,        # soglia rispetto al miglior candidato
        pacer_step: int = 20,
        pacer_delta: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eps = float(exploration_factor)
        self.exploration_decay = float(exploration_decay)
        self.exploration_min = float(exploration_min)
        self.alpha = float(exploration_alpha)
        self.round_threshold = float(round_threshold)
        self.cut_off_util = float(cut_off_util)
        self.pacer_step = int(pacer_step)
        self.pacer_delta = float(pacer_delta)

        # stato per client (chiave: cid)
        self._stat_util: dict[int, float] = {}
        self._duration: dict[int, float] = {}
        self._last_round: dict[int, int] = {}
        self._explored: set[int] = set()
        # storia delle utilita' per il pacer
        self._util_history: list[float] = []

    # ------------------------------------------------------------ selezione
    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []

        self._run_pacer(server_round)

        cid_of = self._node_to_cid
        explored = [n for n in available if cid_of[n] in self._explored]
        unexplored = [n for n in available if cid_of[n] not in self._explored]

        # nessuna informazione ancora: selezione uniforme (round 1). Stessa
        # chiamata al generatore di FedAvg -> con eps=1 le due politiche
        # producono esattamente la stessa sequenza (controllo di regressione).
        if not explored or self.eps >= 1.0:
            idx = self._rng.choice(len(available), size=k, replace=False)
            return [available[i] for i in idx]

        n_explore = min(int(round(self.eps * k)), len(unexplored))
        n_exploit = k - n_explore

        selected: list[int] = []

        # --- sfruttamento: campionamento pesato sopra il cut-off ---
        if n_exploit > 0 and explored:
            T = self._reference_duration()
            utils = np.array(
                [self._utility(cid_of[n], server_round, T) for n in explored],
                dtype=float,
            )
            best = utils.max()
            if best > 0:
                keep = utils >= self.cut_off_util * best
                # se il cut-off e' troppo aggressivo per servire n_exploit
                # client, si allarga ai migliori disponibili
                if keep.sum() < n_exploit:
                    order = np.argsort(-utils)[: max(n_exploit, 1)]
                    keep = np.zeros_like(utils, dtype=bool)
                    keep[order] = True
                cand = [explored[i] for i in np.flatnonzero(keep)]
                w = utils[keep]
                w = w / w.sum()
                take = min(n_exploit, len(cand))
                pick = self._rng.choice(len(cand), size=take, replace=False, p=w)
                selected.extend(cand[i] for i in pick)

        # --- esplorazione: uniforme tra i mai selezionati ---
        n_explore_done = 0
        remaining = k - len(selected)
        if remaining > 0 and unexplored:
            take = min(remaining, len(unexplored))
            pick = self._rng.choice(len(unexplored), size=take, replace=False)
            selected.extend(unexplored[i] for i in pick)
            n_explore_done = take

        # --- riempimento, se cut-off ed esplorazione non bastano ---
        remaining = k - len(selected)
        if remaining > 0:
            pool = [n for n in available if n not in set(selected)]
            if pool:
                take = min(remaining, len(pool))
                pick = self._rng.choice(len(pool), size=take, replace=False)
                selected.extend(pool[i] for i in pick)

        n_exploit_done = len(selected) - n_explore_done
        self.eps = max(self.exploration_min, self.eps * self.exploration_decay)
        log(
            INFO,
            "oort round %s: %s sfruttamento + %s esplorazione | eps=%.3f | "
            "esplorati %s/%s",
            server_round,
            max(0, n_exploit_done),
            n_explore_done,
            self.eps,
            len(self._explored),
            len(cid_of),
        )
        return selected

    # ------------------------------------------------------------- utilita'
    def _utility(self, cid: int, server_round: int, T: float) -> float:
        u = self._stat_util.get(cid, 0.0)

        # staleness: premia chi non partecipa da tempo (Sez. 4.1)
        last = self._last_round.get(cid, 0)
        if server_round > 1 and last >= 1:
            u += math.sqrt(0.1 * math.log(server_round) / last)

        # system utility: penalizza solo chi sfora la durata di riferimento
        t_i = self._duration.get(cid, 0.0)
        if T > 0 and t_i > T:
            u *= (T / t_i) ** self.alpha
        return float(u)

    def _reference_duration(self) -> float:
        if not self._duration:
            return 0.0
        return float(
            np.percentile(list(self._duration.values()), self.round_threshold)
        )

    def _run_pacer(self, server_round: int) -> None:
        """Rilassa il vincolo temporale se l'utilita' raccolta sta calando."""
        if self.pacer_step <= 0 or server_round % self.pacer_step != 0:
            return
        if len(self._util_history) < 2 * self.pacer_step:
            return
        prev = sum(self._util_history[-2 * self.pacer_step : -self.pacer_step])
        curr = sum(self._util_history[-self.pacer_step :])
        if curr < prev:
            self.round_threshold = min(100.0, self.round_threshold + self.pacer_delta)
            log(INFO, "oort pacer: round_threshold -> %.1f", self.round_threshold)

    # --------------------------------------------------------------- update
    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        sq_sum = self._metric(reply, "loss-sq-sum", default=None)
        n = self._metric(reply, "loss-n", default=None)

        if sq_sum is not None and n:
            # Eq. 1: |B| * sqrt( mean(Loss^2) )
            util = float(n_ex) * math.sqrt(float(sq_sum) / float(n))
        else:
            # fallback: senza le statistiche per campione si usa la loss media
            # (approssimazione, non fedele al paper)
            loss = self._metric(reply, "train_loss", default=0.0)
            util = float(n_ex) * abs(float(loss))

        self._stat_util[cid] = util
        self._duration[cid] = float(dt_s)
        self._last_round[cid] = int(server_round)
        self._explored.add(int(cid))
        self._util_history.append(util)