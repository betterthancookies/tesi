from __future__ import annotations

import math
from logging import INFO
import numpy as np
from flwr.app import Message
from flwr.common import log

from device.battery_model import V_NOMINAL
from strategies.fedavg import PhysicalFedAvg


class PhysicalESCS(PhysicalFedAvg):
    def __init__(
        self,
        *args,
        utility_mode: str = "s",
        selection_mode: str = "d",
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
        if not partition_sizes:
            raise ValueError(
                "partition_sizes obbligatorio: senza le taglie reali la "
                "latenza di riferimento e' sbagliata e l'utilita' collassa"
            )
        self.utility_mode, self.selection_mode = um, sm
        self.first_round_all = bool(first_round_all)
        self.w_b, self.w_c, self.w_l = (
            float(battery_weight), float(cpu_cost_weight), float(link_prob_weight)
        )
        self.time_percentile = float(time_percentile)
        self.min_battery = float(min_battery)
        self.min_nq = float(min_network_quality)
        self.cap_prob = bool(cap_probabilistic)
        self._sizes = {int(c): int(n) for c, n in partition_sizes.items()}

        self._loss: dict[int, float] = {}
        self._cons: dict[int, float] = {}
        self._nq: dict[int, float] = {}
        self._lat_ref: dict[int, float] = {}
        self._max_lat = 0.0
        self._lim = 1.0
        self._cap_wh: dict[int, float] = {}

    @property
    def variant(self) -> str:
        return f"escs-{self.utility_mode}{self.selection_mode}-{self.world.battery}"

    def _ensure_profile(self) -> None:
        if self._lat_ref:
            return
        for cid in sorted(self._node_to_cid.values()):
            if cid not in self._sizes:
                raise KeyError(f"partition_sizes non contiene il cid {cid}")
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
        return float(self.world.snapshot(cid).soc)

    def _consumption(self, cid: int) -> float:
        if cid in self._cons:
            return self._cons[cid]
        return float(np.mean(list(self._cons.values()))) if self._cons else 0.0

    def _has_resources(self, cid: int) -> bool:
        return (self._battery(cid) - self._consumption(cid) >= self.min_battery
                and self._nq[cid] >= self.min_nq)

    def _select_clients(
        self, available: list[int], k: int, server_round: int
    ) -> list[int]:
        if not k:
            return []
        self._ensure_profile()
        cid_of = self._node_to_cid

        if server_round == 1 and self.first_round_all:
            log(INFO, "%s round 1: tutti i %s disponibili", self.variant,
                len(available))
            return list(available)

        ast = [n for n in available if self._has_resources(cid_of[n])]

        max_loss = max(self._loss.values()) if self._loss else 1.0
        util = {n: self._utility(cid_of[n], max_loss) for n in ast}
        ast = [ast[i] for i in self._rng.permutation(len(ast))] if ast else []

        frac = self.k_per_round / max(1, len(cid_of))
        rate = float("nan")
        selected: list[int] = []
        if ast and self.selection_mode == "p":
            selected = [n for n in ast if self._rng.random() <= util[n]]
            rate = len(selected) / len(ast)
            if self.cap_prob and len(selected) > k:
                idx = self._rng.choice(len(selected), size=k, replace=False)
                selected = [selected[i] for i in idx]

        if not selected and ast:
            ranked = sorted(ast, key=lambda n: -util[n])
            if self.selection_mode == "d":
                selected = ast if len(ast) <= k else ranked[:k]
            else:
                selected = ranked[:round(len(ast) * frac)]

        if not selected:
            batt = [n for n in available
                    if self._battery(cid_of[n]) - self._consumption(cid_of[n])
                    >= self.min_battery]
            batt.sort(key=lambda n: -self._loss.get(cid_of[n], max_loss))
            selected = batt[:round(len(batt) * frac)]

        log(INFO, "%s round %s: %s idonei su %s, %s selezionati (accept %.2f)",
            self.variant, server_round, len(ast), len(available),
            len(selected), rate)
        return selected

    def _utility(self, cid: int, max_loss: float) -> float:
        bat = self._battery(cid)
        cpu = self._lat_ref[cid] / self._max_lat
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

    def _on_round_result(
        self, cid: int, n_ex: int, dt_s: float, e_wh: float, e_lin_wh: float,
        reply: Message, server_round: int,
    ) -> None:
        cap = self._capacity(cid)
        if cap > 0.0:
            self._cons[cid] = float(e_wh / cap)
        if n_ex > 0:
            loss = self._metric(reply, "train_loss", "loss", default=None)
            if loss is not None and math.isfinite(float(loss)):
                self._loss[cid] = float(loss)
