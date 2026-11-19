from __future__ import annotations

import numpy as np

from device.constants import WORKLOAD_CHAINS, WORKLOAD_STATES

STATE_NAMES = list(WORKLOAD_STATES)
STATE_UTIL = np.array(list(WORKLOAD_STATES.values()), dtype=float)
PROFILE_NAMES = list(WORKLOAD_CHAINS)
W_MAX = 0.95


def scaled_util(scale: float) -> np.ndarray:
    if scale <= 0.0:
        raise ValueError(f"workload scale {scale}: deve essere > 0")
    return np.minimum(W_MAX, STATE_UTIL * float(scale))


class WorkloadModel:
    def __init__(self, n_clients: int, seed: int, enabled: bool = True,
                 scale: float = 1.0) -> None:
        self.enabled = bool(enabled)
        self.scale = float(scale)
        self._util = scaled_util(self.scale)
        self.n_clients = int(n_clients)
        self._rng = np.random.default_rng(seed)
        self._chains = {p: np.asarray(WORKLOAD_CHAINS[p], dtype=float)
                        for p in PROFILE_NAMES}
        for p, m in self._chains.items():
            if not np.allclose(m.sum(axis=1), 1.0):
                raise ValueError(f"la catena '{p}' ha righe che non sommano a 1")

        self._profile = self._assign_equal(PROFILE_NAMES)
        self._state = {cid: 0 for cid in range(self.n_clients)}

    def _assign_equal(self, names: list[str]) -> dict[int, str]:
        base = [names[i % len(names)] for i in range(self.n_clients)]
        order = self._rng.permutation(self.n_clients)
        return {int(cid): base[i] for i, cid in enumerate(order)}

    def step(self) -> None:
        if not self.enabled:
            return
        for cid in range(self.n_clients):
            row = self._chains[self._profile[cid]][self._state[cid]]
            self._state[cid] = int(self._rng.choice(len(row), p=row))

    def utilization(self, cid: int) -> float:
        if not self.enabled:
            return 0.0
        return float(self._util[self._state[cid]])

    def max_utilization(self) -> float:
        return float(self._util.max())

    def profile(self, cid: int) -> str:
        return self._profile[cid] if self.enabled else "disabled"

    def state(self, cid: int) -> str:
        return STATE_NAMES[self._state[cid]] if self.enabled else "idle"

    def mean_utilization(self) -> float:
        if not self.enabled:
            return 0.0
        return float(np.mean([self.utilization(c) for c in range(self.n_clients)]))

    def counts_by_profile(self) -> dict[str, int]:
        out = {p: 0 for p in PROFILE_NAMES}
        for cid in range(self.n_clients):
            out[self._profile[cid]] += 1
        return out


def stationary_utilization(scale: float = 1.0) -> dict[str, float]:
    util = scaled_util(scale)
    out = {}
    for name, m in WORKLOAD_CHAINS.items():
        m = np.asarray(m, dtype=float)
        vals, vecs = np.linalg.eig(m.T)
        i = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, i])
        pi = pi / pi.sum()
        out[name] = float(pi @ util)
    return out


if __name__ == "__main__":
    print("utilizzazione media attesa (dalla distribuzione stazionaria):")
    for k, v in stationary_utilization().items():
        print(f"  {k:>7}: {v:.4f}")

    print("\nverifica per simulazione (30 client, 20000 round):")
    wl = WorkloadModel(n_clients=30, seed=2026)
    acc = {p: [] for p in PROFILE_NAMES}
    for _ in range(20000):
        wl.step()
        for cid in range(30):
            acc[wl.profile(cid)].append(wl.utilization(cid))
    for p, v in acc.items():
        print(f"  {p:>7}: {np.mean(v):.4f}  (n={len(v)})")
    print("\nripartizione dei profili:", wl.counts_by_profile())
