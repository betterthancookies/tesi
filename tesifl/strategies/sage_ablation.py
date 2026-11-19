from __future__ import annotations

from logging import INFO

import numpy as np
import pulp
from flwr.app import ConfigRecord, Message
from flwr.common import log

from strategies.fedavg import PhysicalFedAvg
from strategies.sage import _js


class PhysicalSAGEAblation(PhysicalFedAvg):
    def __init__(
        self,
        *args,
        sage_a: float = 0.5,
        sage_b: float = 0.5,
        sage_k: int = 2,
        half_epochs_threshold: float = 0.3,
        max_fraction: float = 0.2,
        class_distributions: dict[int, np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.a, self.b = float(sage_a), float(sage_b)
        if abs(self.a + self.b - 1.0) > 1e-9:
            raise ValueError("a + b = 1")
        self.k = int(sage_k)
        self.half_th = float(half_epochs_threshold)
        self.max_frac = float(max_fraction)

        self._dist: dict[int, np.ndarray] = {}
        self._div: dict[int, float] = {}
        self._half: set[int] = set()

        if class_distributions:
            for cid, d in class_distributions.items():
                arr = np.asarray(d, dtype=float)
                self._dist[int(cid)] = arr / arr.sum()
            self._recompute_div()

    def _resource(self, cid: int) -> float:
        return float(self.world.snapshot(cid).soc)

    def _divergence(self, cid: int) -> float:
        if cid in self._div:
            return self._div[cid]
        return float(np.mean(list(self._div.values()))) if self._div else 0.0

    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k or not available:
            log(INFO, "sage-soc round %s: pool vuoto", server_round)
            return []
        cid_of = self._node_to_cid

        pool = available
        m = len(pool)
        hi = min(self.k + server_round - 1,
                 int(np.floor(self.max_frac * max(m, self._n_alive))), m)
        lo = self.k

        score = {
            n: self.a * self._resource(cid_of[n])
            + self.b * self._divergence(cid_of[n])
            for n in pool
        }

        prob = pulp.LpProblem("SAGE_SoC_Select", pulp.LpMaximize)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]
        prob += pulp.lpSum(y[i] * score[pool[i]] for i in range(m))
        prob += pulp.lpSum(y) >= lo
        prob += pulp.lpSum(y) <= hi
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [pool[i] for i in range(m) if int(pulp.value(y[i]) or 0) == 1]
        if not selected:
            log(INFO, "sage-soc round %s: ILP infeasible con pool %s "
                "(lo=%s, hi=%s): pool esaurito, la run e' finita",
                server_round, m, lo, hi)
            return []

        self._mark_half(selected)
        log(INFO, "sage-soc round %s: pool %s su %s, %s selezionati "
            "(cap %s, %s a epoche dimezzate)", server_round, m,
            len(self._node_to_cid), len(selected), hi, len(self._half))
        return selected

    def _mark_half(self, selected: list[int]) -> None:
        self._half = {
            self._node_to_cid[n] for n in selected
            if self._resource(self._node_to_cid[n]) < self.half_th
        }

    def _per_client_config(
        self, cid: int, cfg: ConfigRecord, server_round: int
    ) -> None:
        if cid in self._half:
            cfg["epochs"] = max(1, int(self.epochs) // 2)

    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        dist = self._metric(reply, "class-dist", default=None)
        if dist is None:
            return
        arr = np.asarray(list(dist), dtype=float)
        if arr.sum() <= 0:
            return
        self._dist[cid] = arr / arr.sum()
        self._recompute_div()

    def _recompute_div(self) -> None:
        cids = sorted(self._dist)
        n = len(cids)
        if n < 2:
            self._div = {c: 0.0 for c in cids}
            return
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                mat[i, j] = mat[j, i] = _js(self._dist[cids[i]], self._dist[cids[j]])
        self._div = {c: float(mat[i].sum() / (n - 1)) for i, c in enumerate(cids)}
