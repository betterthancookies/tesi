from __future__ import annotations

from dataclasses import dataclass

from device.communication_model import comm_power_w, comm_time_s
from device.compute_model import probe_time_s, training_time_s
from device.device_profile import DeviceProfile
from device.world_state import (
    BATTERIES, BATTERY_NAMES, WorldState, battery_delta_soc,
    effective_profile, training_power_w,
)

RECORD_KEY = "device"
MODELS = BATTERIES
_NAMES = BATTERY_NAMES


@dataclass(frozen=True)
class DeviceReading:
    soc: float
    workload: float
    charging: bool
    profile: DeviceProfile
    model: str = "nm"

    @classmethod
    def from_world(cls, world: WorldState, cid: int) -> "DeviceReading":
        return cls(
            soc=float(world.snapshot(cid).soc),
            workload=float(world.utilization(cid)),
            charging=bool(world.is_charging(cid)),
            profile=world.profile(cid),
            model=world.battery,
        )

    def to_record(self) -> dict:
        p = self.profile
        return {
            "soc": float(self.soc),
            "workload": float(self.workload),
            "charging": bool(self.charging),
            "battery-model": _NAMES[self.model],
            "cid": int(p.cid),
            "capacity-mah": float(p.battery_capacity_mah),
            "macs-per-s": float(p.macs_per_s),
            "train-power-w": float(p.train_power_w),
            "peukert-n": float(p.peukert_n),
            "tier": str(p.tier),
        }

    @classmethod
    def from_record(cls, rec) -> "DeviceReading":
        return cls(
            soc=float(rec["soc"]),
            workload=float(rec["workload"]),
            charging=bool(rec["charging"]),
            profile=DeviceProfile(
                cid=int(rec["cid"]),
                battery_capacity_mah=float(rec["capacity-mah"]),
                macs_per_s=float(rec["macs-per-s"]),
                peukert_n=float(rec["peukert-n"]),
                tier=str(rec["tier"]),
                train_power_w=float(rec["train-power-w"]),
            ),
            model={v: k for k, v in _NAMES.items()}[str(rec["battery-model"])],
        )

    def work_time_s(self, epochs: float, batch_size: int, n_samples: int,
                    probe_samples: int = 0) -> float:
        p = effective_profile(self.profile, self.workload)
        return (training_time_s(p, epochs, n_samples, batch_size)
                + probe_time_s(p, probe_samples))

    def soc_after(self, dt_s: float, batch_size: int) -> tuple[float, float]:
        if self.charging:
            return self.soc, self.soc
        s1 = self._drain(self.soc, training_power_w(self.profile), dt_s)
        if s1 <= 0.0:
            return 0.0, 0.0
        s2 = self._drain(s1, comm_power_w(), comm_time_s())
        return s1, s2

    def predict(self, epochs: float, batch_size: int, n_samples: int,
                probe_samples: int = 0) -> tuple[float, float]:
        dt = self.work_time_s(epochs, batch_size, n_samples, probe_samples)
        return dt, self.soc_after(dt, batch_size)[1]

    def _drain(self, soc: float, power_w: float, dt_s: float) -> float:
        d = battery_delta_soc(self.model, power_w, dt_s, self.profile, soc)
        return max(0.0, min(1.0, soc + d))
