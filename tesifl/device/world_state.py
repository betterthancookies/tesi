from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from device.battery_model import (
    delta_soc, energy_wh_from_delta_soc, energy_wh_linear,
)
from device.battery_model_peukert import delta_soc as peukert_delta_soc
from device.communication_model import (
    comm_power_w, comm_time_s, ctrl_power_w, ctrl_time_s, download_power_w,
    download_time_s,
)
from device.compute_model import probe_time_s, training_time_s
from device.constants import (
    DCDC_EFFICIENCY,
    CHARGE_C_RATE,
    IDLE_POWER_W,
    INITIAL_SOC_MEAN,
    INITIAL_SOC_RANGE,
    INITIAL_SOC_STD,
    LOW_BATTERY_SOC,
    RECHARGE_AVAILABLE_DEFAULT,
    RECHARGE_RATE_PER_H,
    V_NOMINAL,
)
from device.device_profile import DeviceProfile, generate_profiles
from device.workload import WorkloadModel


def effective_profile(p: DeviceProfile, w: float) -> DeviceProfile:
    if w <= 0.0:
        return p
    return DeviceProfile(
        cid=p.cid, battery_capacity_mah=p.battery_capacity_mah,
        macs_per_s=p.macs_per_s * max(1e-3, 1.0 - w),
        peukert_n=p.peukert_n, tier=p.tier, train_power_w=p.train_power_w,
    )


def training_power_w(p: DeviceProfile) -> float:
    return p.train_power_w


def idle_power_w(p: DeviceProfile, w: float) -> float:
    return IDLE_POWER_W + w * (p.train_power_w - IDLE_POWER_W)


BATTERIES = ("lin", "peuk", "nm")
BATTERY_NAMES = {"lin": "linear", "peuk": "Peukert", "nm": "datasheet"}


def battery_delta_soc(battery: str, power_w: float, dt_s: float,
                      profile: DeviceProfile, soc: float) -> float:
    if battery == "nm":
        return delta_soc(power_w, dt_s, profile, soc)
    if battery == "peuk":
        return peukert_delta_soc(power_w, dt_s, profile)
    if battery == "lin":
        cap_wh = profile.battery_capacity_mah / 1000.0 * V_NOMINAL
        return -energy_wh_linear(power_w, dt_s) / cap_wh
    raise ValueError(f"batteria '{battery}': una fra {BATTERIES}")


@dataclass(frozen=True)
class WorldStateSnapshot:
    cid: int
    soc: float
    available: bool
    recharging: bool
    failed: bool


class WorldState:
    def __init__(self, profiles: list[DeviceProfile], seed: int,
                 workload: WorkloadModel | None = None,
                 idle_enabled: bool = True,
                 recharge_enabled: bool = False,
                 recharge_available: bool = RECHARGE_AVAILABLE_DEFAULT,
                 battery: str = "nm") -> None:
        if battery not in BATTERIES:
            raise ValueError(f"batteria '{battery}': una fra {BATTERIES}")
        self.battery = battery
        self._profiles = {p.cid: p for p in profiles}
        self._rng = np.random.default_rng(seed)
        self._idle_enabled = bool(idle_enabled)
        self._recharge_enabled = bool(recharge_enabled)
        self._recharge_available = bool(recharge_available)
        self.workload = workload if workload is not None else WorkloadModel(
            n_clients=len(profiles), seed=seed + 7919, enabled=False)
        self._soc: dict[int, float] = {
            cid: self._sample_initial_soc() for cid in self._profiles
        }
        self._soc0: dict[int, float] = dict(self._soc)
        self._failed: dict[int, bool] = {cid: False for cid in self._profiles}
        self._energy_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_comm_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_ctrl_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_lin_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_idle_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._energy_grid_wh: dict[int, float] = {cid: 0.0 for cid in self._profiles}
        self._charging: dict[int, bool] = {cid: False for cid in self._profiles}
        self._n_recharges: dict[int, int] = {cid: 0 for cid in self._profiles}
        self._last_round_s = 0.0

    def _sample_initial_soc(self) -> float:
        lo, hi = INITIAL_SOC_RANGE
        for _ in range(10000):
            x = float(self._rng.normal(INITIAL_SOC_MEAN, INITIAL_SOC_STD))
            if lo <= x <= hi:
                return x
        raise RuntimeError(
            f"SoC iniziale: nessun campione in [{lo}, {hi}] con "
            f"mu={INITIAL_SOC_MEAN}, sigma={INITIAL_SOC_STD}"
        )

    def snapshot(self, cid: int) -> WorldStateSnapshot:
        failed = self._failed[cid]
        charging = self._charging[cid]
        available = (not failed) and (self._recharge_available or not charging)
        return WorldStateSnapshot(
            cid=cid, soc=self._soc[cid], available=available,
            recharging=charging, failed=failed,
        )

    def profile(self, cid: int) -> DeviceProfile:
        return self._profiles[cid]

    def initial_soc(self, cid: int) -> float:
        return self._soc0[cid]

    def capacity_wh(self, cid: int) -> float:
        return self._profiles[cid].battery_capacity_mah / 1000.0 * V_NOMINAL

    def soc_vector(self) -> list[float]:
        return [float(self._soc[c]) for c in sorted(self._profiles)]

    def device_rows(self) -> list[dict]:
        rows = []
        for c in sorted(self._profiles):
            p = self._profiles[c]
            rows.append({
                "cid": c, "tier": p.tier, "capacity_mah": p.battery_capacity_mah,
                "workload_chain": getattr(self.workload, "_profile", {}).get(c, ""),
                "soc0": self._soc0[c], "soc_final": self._soc[c],
                "depleted": int(self._failed[c]), "recharges": self._n_recharges[c],
                "energy_train_wh": self._energy_wh[c],
                "energy_comm_wh": self._energy_comm_wh[c],
                "energy_ctrl_wh": self._energy_ctrl_wh[c],
                "energy_idle_wh": self._energy_idle_wh[c],
                "energy_grid_wh": self._energy_grid_wh[c],
                "energy_lin_wh": self._energy_lin_wh[c],
            })
        return rows

    def energy_wh(self, cid: int) -> float:
        return (self._energy_wh[cid] + self._energy_comm_wh[cid]
                + self._energy_ctrl_wh[cid])

    def energy_lin_wh(self, cid: int) -> float:
        return self._energy_lin_wh[cid]

    def energy_idle_wh(self, cid: int) -> float:
        return self._energy_idle_wh[cid]

    def tier(self, cid: int) -> str:
        return self._profiles[cid].tier

    def utilization(self, cid: int) -> float:
        return self.workload.utilization(cid)

    def is_charging(self, cid: int) -> bool:
        return self._charging[cid]

    def energy_grid_wh(self, cid: int) -> float:
        return self._energy_grid_wh[cid]

    def begin_round(self) -> None:
        self.workload.step()
        if not self._recharge_enabled:
            return
        dt_h = self._last_round_s / 3600.0
        for cid in self._profiles:
            if self._failed[cid] or self._charging[cid]:
                continue
            p = 1.0 - np.exp(-RECHARGE_RATE_PER_H * (1.0 - self._soc[cid]) * dt_h)
            if self._rng.random() < p:
                self._charging[cid] = True
                self._n_recharges[cid] += 1

    def tick_all(self, dt_s: float = 0.0) -> None:
        return None

    def effective_profile(self, cid: int) -> DeviceProfile:
        return effective_profile(self._profiles[cid],
                                 self.workload.utilization(cid))

    def round_duration_s(
        self, cid: int, epochs: int, batch_size: int, dataset_size_local: int,
        include_workload: bool = True,
    ) -> float:
        p = self.effective_profile(cid) if include_workload else self._profiles[cid]
        return training_time_s(p, epochs, dataset_size_local, batch_size)

    def probe_duration_s(self, cid: int, n_samples: int) -> float:
        return probe_time_s(self.effective_profile(cid), n_samples)

    def communication_time_s(self, cid: int) -> float:
        return comm_time_s()

    def training_power_w(self, cid: int, batch_size: int | None = None) -> float:
        return training_power_w(self._profiles[cid])

    def _drain(self, cid: int, power_w: float, dt_s: float) -> tuple[float, float]:
        if self._failed[cid]:
            return 0.0, 0.0
        if self._charging[cid]:
            self._energy_grid_wh[cid] += (power_w / DCDC_EFFICIENCY) * dt_s / 3600.0
            return 0.0, 0.0
        profile = self._profiles[cid]
        soc = self._soc[cid]
        d_req = battery_delta_soc(self.battery, power_w, dt_s, profile, soc)
        new_soc = max(0.0, min(1.0, soc + d_req))
        d = new_soc - soc
        self._soc[cid] = new_soc
        if new_soc <= 0.0:
            self._failed[cid] = True
        frac = d / d_req if d_req < 0.0 else 0.0
        e_lin = energy_wh_linear(power_w, dt_s) * frac
        self._energy_lin_wh[cid] += e_lin
        return energy_wh_from_delta_soc(d, profile), e_lin

    def apply_round(self, cid: int, dt_s: float, batch_size: int) -> tuple[float, float]:
        e, e_lin = self._drain(cid, self.training_power_w(cid, batch_size), dt_s)
        self._energy_wh[cid] += e
        return e, e_lin

    def apply_control(self, cid: int, merged: bool = False) -> tuple[float, float]:
        e, e_lin = self._drain(cid, ctrl_power_w(merged), ctrl_time_s(merged))
        self._energy_ctrl_wh[cid] += e
        return e, e_lin

    def apply_communication(self, cid: int) -> tuple[float, float]:
        e, e_lin = self._drain(cid, comm_power_w(), comm_time_s())
        self._energy_comm_wh[cid] += e
        return e, e_lin

    def apply_download(self, cid: int) -> tuple[float, float]:
        e, e_lin = self._drain(cid, download_power_w(), download_time_s())
        self._energy_comm_wh[cid] += e
        return e, e_lin

    def idle_power_w(self, cid: int) -> float:
        return idle_power_w(self._profiles[cid], self.workload.utilization(cid))

    def apply_idle(self, durations: dict[int, float]) -> tuple[float, float]:
        if not self._idle_enabled:
            return 0.0, 0.0
        t_round = max(durations.values()) if durations else 0.0
        if t_round <= 0.0:
            return 0.0, 0.0
        self._last_round_s = t_round
        self._charge_step(t_round)
        tot_e = tot_lin = 0.0
        for cid in self._profiles:
            if self._failed[cid]:
                continue
            idle_s = t_round - float(durations.get(cid, 0.0))
            if idle_s <= 0.0:
                continue
            e, lin = self._drain(cid, self.idle_power_w(cid), idle_s)
            self._energy_idle_wh[cid] += e
            tot_e += e
            tot_lin += lin
        return tot_e, tot_lin

    def _charge_step(self, t_round: float) -> None:
        if not self._recharge_enabled:
            return
        d_soc = CHARGE_C_RATE * t_round / 3600.0
        for cid in self._profiles:
            if not self._charging[cid]:
                continue
            self._soc[cid] = min(1.0, self._soc[cid] + d_soc)
            if self._soc[cid] >= 1.0:
                self._charging[cid] = False

    def total_energy_train_wh(self) -> float:
        return sum(self._energy_wh.values())

    def total_energy_comm_wh(self) -> float:
        return sum(self._energy_comm_wh.values())

    def total_energy_grid_wh(self) -> float:
        return sum(self._energy_grid_wh.values())

    def total_energy_idle_wh(self) -> float:
        return sum(self._energy_idle_wh.values())

    def total_energy_ctrl_wh(self) -> float:
        return sum(self._energy_ctrl_wh.values())

    def total_energy_wh(self) -> float:
        return (self.total_energy_train_wh() + self.total_energy_comm_wh()
                + self.total_energy_ctrl_wh() + self.total_energy_idle_wh())

    def total_energy_lin_wh(self) -> float:
        return sum(self._energy_lin_wh.values())

    def stats(self) -> dict:
        soc_all = list(self._soc.values())
        return {
            "total_energy_wh": self.total_energy_wh(),
            "energy_train_wh": self.total_energy_train_wh(),
            "energy_comm_wh": self.total_energy_comm_wh(),
            "energy_ctrl_wh": self.total_energy_ctrl_wh(),
            "energy_idle_wh": self.total_energy_idle_wh(),
            "energy_grid_wh": sum(self._energy_grid_wh.values()),
            "energy_lin_wh": self.total_energy_lin_wh(),
            "mean_workload": self.workload.mean_utilization(),
            "n_failed": sum(self._failed.values()),
            "n_recharging": int(sum(1 for v in self._charging.values() if v)),
            "total_recharges": int(sum(self._n_recharges.values())),
            "mean_soc": float(np.mean(soc_all)),
            "median_soc": float(np.median(soc_all)),
            "min_soc": float(np.min(soc_all)),
            "max_soc": float(np.max(soc_all)),
            "p25_soc": float(np.percentile(soc_all, 25)),
            "p75_soc": float(np.percentile(soc_all, 75)),
            "n_soc_zero": int(sum(1 for v in soc_all if v <= 0.0)),
            "n_soc_low": int(sum(1 for v in soc_all if v < LOW_BATTERY_SOC)),
            "n_clients": len(soc_all),
            **{f"soc_{t}": float(np.mean([self._soc[c] for c in self._profiles
                                          if self._profiles[c].tier == t]))
               for t in sorted({p.tier for p in self._profiles.values()})},
            **{f"dead_{t}": int(sum(1 for c in self._profiles
                                    if self._profiles[c].tier == t
                                    and self._soc[c] <= 0.0))
               for t in sorted({p.tier for p in self._profiles.values()})},
        }


def build_world_state(n_clients: int, seed: int,
                      capacity_mah: float | None = None,
                      tiers_enabled: bool = True,
                      workload_enabled: bool = True,
                      idle_enabled: bool = True,
                      recharge_enabled: bool = False,
                      recharge_available: bool = True,
                      battery: str = "nm",
                      workload_scale: float = 1.0,
                      **kwargs) -> WorldState:
    profiles = generate_profiles(n_clients, seed=seed, capacity_mah=capacity_mah,
                                 tiers_enabled=tiers_enabled)
    workload = WorkloadModel(n_clients=n_clients, seed=seed + 7919,
                             enabled=workload_enabled, scale=workload_scale)
    return WorldState(profiles=profiles, seed=seed, workload=workload,
                      idle_enabled=idle_enabled,
                      recharge_enabled=recharge_enabled,
                      recharge_available=recharge_available,
                      battery=battery)


if __name__ == "__main__":
    N, K, R = 30, 6, 250
    ws = build_world_state(n_clients=N, seed=42)
    rng = np.random.default_rng(0)
    for r in range(1, R + 1):
        alive = [c for c in range(N) if ws.snapshot(c).available]
        if not alive:
            print(f"popolazione esaurita al round {r}")
            break
        sel = rng.choice(alive, size=min(K, len(alive)), replace=False)
        for cid in sel:
            dt = ws.round_duration_s(cid, epochs=5, batch_size=128,
                                     dataset_size_local=1333)
            ws.apply_round(cid, dt_s=dt, batch_size=128)
            ws.apply_communication(cid)
        if r % 50 == 0:
            s = ws.stats()
            print(f"round {r:3d} | {s['total_energy_wh']:6.3f} Wh "
                  f"(train {s['energy_train_wh']:.3f} + comm {s['energy_comm_wh']:.3f}) "
                  f"| SoC medio {s['mean_soc']:.3f} | morti {s['n_soc_zero']}/{N}")
