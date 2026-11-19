from __future__ import annotations

import math
from logging import INFO

import numpy as np
from flwr.app import Message
from flwr.common import log

from strategies.fedavg import PhysicalFedAvg


class PhysicalOort(PhysicalFedAvg):
    def __init__(
        self,
        *args,
        exploration_factor: float = 0.9,
        exploration_decay: float = 0.98,
        exploration_min: float = 0.3,
        exploration_alpha: float = 0.3,
        round_threshold: float = 30.0,
        cut_off_util: float = 0.95,
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

        self._stat_util: dict[int, float] = {}
        self._duration: dict[int, float] = {}
        self._last_round: dict[int, int] = {}
        self._explored: set[int] = set()
        self._util_history: list[float] = []

    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []

        self._run_pacer(server_round)

        cid_of = self._node_to_cid
        explored = [n for n in available if cid_of[n] in self._explored]
        unexplored = [n for n in available if cid_of[n] not in self._explored]

        if not explored or self.eps >= 1.0:
            idx = self._rng.choice(len(available), size=k, replace=False)
            return [available[i] for i in idx]

        n_explore = min(int(round(self.eps * k)), len(unexplored))
        n_exploit = k - n_explore

        selected: list[int] = []

        if n_exploit > 0 and explored:
            T = self._reference_duration()
            utils = np.array(
                [self._utility(cid_of[n], server_round, T) for n in explored],
                dtype=float,
            )
            best = utils.max()
            if best > 0:
                keep = utils >= self.cut_off_util * best
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

        n_explore_done = 0
        remaining = k - len(selected)
        if remaining > 0 and unexplored:
            take = min(remaining, len(unexplored))
            pick = self._rng.choice(len(unexplored), size=take, replace=False)
            selected.extend(unexplored[i] for i in pick)
            n_explore_done = take

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

    def _utility(self, cid: int, server_round: int, T: float) -> float:
        u = self._stat_util.get(cid, 0.0)

        last = self._last_round.get(cid, 0)
        if server_round > 1 and last >= 1:
            u += math.sqrt(0.1 * math.log(server_round) / last)

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
        if self.pacer_step <= 0 or server_round % self.pacer_step != 0:
            return
        if len(self._util_history) < 2 * self.pacer_step:
            return
        prev = sum(self._util_history[-2 * self.pacer_step : -self.pacer_step])
        curr = sum(self._util_history[-self.pacer_step :])
        if curr < prev:
            self.round_threshold = min(100.0, self.round_threshold + self.pacer_delta)
            log(INFO, "oort pacer: round_threshold -> %.1f", self.round_threshold)

    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        sq_sum = self._metric(reply, "loss-sq-sum", default=None)
        n = self._metric(reply, "loss-n", default=None)

        if sq_sum is not None and n:
            util = float(n_ex) * math.sqrt(float(sq_sum) / float(n))
        else:
            loss = self._metric(reply, "train_loss", default=0.0)
            util = float(n_ex) * abs(float(loss))

        self._stat_util[cid] = util
        self._duration[cid] = float(dt_s)
        self._last_round[cid] = int(server_round)
        self._explored.add(int(cid))
        self._util_history.append(util)
