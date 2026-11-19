from __future__ import annotations

from logging import INFO

import numpy as np
import pulp
from flwr.app import ConfigRecord, Message
from flwr.common import log

from strategies.fedavg import PhysicalFedAvg


def _js(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-9
    p, q = (p + eps) / (p + eps).sum(), (q + eps) / (q + eps).sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * (np.log(a) - np.log(b))) / np.log(2.0))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


class PhysicalSAGE(PhysicalFedAvg):
    def __init__(
        self,
        *args,
        sage_a: float = 0.5,
        sage_b: float = 0.2,
        sage_c: float = 0.3,
        sage_k: int = 2,
        eligibility_threshold: float = 0.2,
        half_epochs_threshold: float = 0.3,
        max_fraction: float = 0.2,
        class_distributions: dict[int, np.ndarray] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.a, self.b, self.c = float(sage_a), float(sage_b), float(sage_c)
        if abs(self.a + self.b + self.c - 1.0) > 1e-9:
            raise ValueError("vincolo (2g): a + b + c = 1")
        self.k = int(sage_k)
        self.elig_th = float(eligibility_threshold)
        self.half_th = float(half_epochs_threshold)
        self.max_frac = float(max_fraction)

        self._dist: dict[int, np.ndarray] = {}
        self._div: dict[int, float] = {}
        self._renew: dict[int, float] = {}
        self._half: set[int] = set()

        if class_distributions:
            for cid, d in class_distributions.items():
                arr = np.asarray(d, dtype=float)
                self._dist[int(cid)] = arr / arr.sum()
            self._recompute_div()

    def _energy(self, cid: int) -> float:
        return float(self.world.snapshot(cid).soc)

    def _divergence(self, cid: int) -> float:
        if cid in self._div:
            return self._div[cid]
        return float(np.mean(list(self._div.values()))) if self._div else 0.0

    def _step_renewable(self) -> None:
        for cid in self._node_to_cid.values():
            cur = self._renew.get(cid)
            if cur is None:
                cur = float(self._rng.uniform(0.0, 1.0))
            self._renew[cid] = float(
                np.clip(cur + self._rng.uniform(-0.1, 0.1), 0.0, 1.0)
            )

    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []
        cid_of = self._node_to_cid
        self._step_renewable()

        elig = [n for n in available if self._energy(cid_of[n]) >= self.elig_th]
        if not elig:
            log(INFO, "sage round %s: nessun eleggibile (soglia %.2f)",
                server_round, self.elig_th)
            return []

        m = len(elig)
        n_total = len(cid_of)
        hi = min(self.k + server_round - 1, int(np.floor(self.max_frac * n_total)), m)
        lo = min(self.k, hi)

        score = {
            n: self.a * self._energy(cid_of[n])
            + self.b * self._divergence(cid_of[n])
            + self.c * self._renew.get(cid_of[n], 0.0)
            for n in elig
        }

        prob = pulp.LpProblem("SAGE_Select", pulp.LpMaximize)
        y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(m)]
        prob += pulp.lpSum(y[i] * score[elig[i]] for i in range(m))
        prob += pulp.lpSum(y) >= lo
        prob += pulp.lpSum(y) <= hi
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        selected = [elig[i] for i in range(m) if int(pulp.value(y[i]) or 0) == 1]
        if not selected:
            selected = sorted(elig, key=lambda n: -score[n])[:lo]

        self._mark_half(selected)
        log(INFO, "sage round %s: %s eleggibili su %s, %s selezionati "
            "(cap %s, %s a epoche dimezzate)", server_round, m, len(available),
            len(selected), hi, len(self._half))
        return selected

    def _mark_half(self, selected: list[int]) -> None:
        self._half = {
            self._node_to_cid[n] for n in selected
            if self._energy(self._node_to_cid[n]) < self.half_th
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
