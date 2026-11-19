from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from device.constants import (
    BATTERY_CAPACITY_MAH,
    MACS_PER_S_UNIFORM,
    PEUKERT_N,
    TIERS,
    TRAIN_POWER_W_UNIFORM,
)

TIER_NAMES = list(TIERS)


@dataclass(frozen=True)
class DeviceProfile:
    cid: int
    battery_capacity_mah: float
    macs_per_s: float
    peukert_n: float
    tier: str = "uniform"
    train_power_w: float = TRAIN_POWER_W_UNIFORM


def generate_profiles(
    n_clients: int,
    seed: int | None = None,
    capacity_mah: float | None = None,
    tiers_enabled: bool = True,
) -> list[DeviceProfile]:
    if not tiers_enabled:
        cap = BATTERY_CAPACITY_MAH if capacity_mah is None else float(capacity_mah)
        return [
            DeviceProfile(cid=cid, battery_capacity_mah=cap,
                          macs_per_s=MACS_PER_S_UNIFORM, peukert_n=PEUKERT_N,
                          tier="uniform", train_power_w=TRAIN_POWER_W_UNIFORM)
            for cid in range(n_clients)
        ]

    rng = np.random.default_rng(0 if seed is None else seed)
    base = [TIER_NAMES[i % len(TIER_NAMES)] for i in range(n_clients)]
    order = rng.permutation(n_clients)
    tier_of = {int(cid): base[i] for i, cid in enumerate(order)}

    profiles = []
    for cid in range(n_clients):
        t = TIERS[tier_of[cid]]
        profiles.append(DeviceProfile(
            cid=cid,
            battery_capacity_mah=float(t["capacity_mah"]),
            macs_per_s=float(t["macs_per_s"]),
            peukert_n=PEUKERT_N,
            tier=tier_of[cid],
            train_power_w=float(t["train_power_w"]),
        ))
    return profiles


def save_profiles(profiles: list[DeviceProfile], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([asdict(p) for p in profiles], f, indent=2)


def load_profiles(path: str | Path) -> list[DeviceProfile]:
    with Path(path).open() as f:
        return [DeviceProfile(**p) for p in json.load(f)]


if __name__ == "__main__":
    from collections import Counter

    profs = generate_profiles(n_clients=30, seed=2026)
    print("ripartizione dei tier:", Counter(p.tier for p in profs))
    for name in TIER_NAMES:
        p = next(x for x in profs if x.tier == name)
        print(f"  {name:>5}: {p.battery_capacity_mah:.0f} mAh, "
              f"{p.macs_per_s / 1e9:.2f} GMAC/s, {p.train_power_w:.2f} W")
    print("\nprimi 12 cid:", [(p.cid, p.tier) for p in profs[:12]])
    print("\nablazione (tiers_enabled=False):",
          generate_profiles(3, seed=2026, tiers_enabled=False)[0])
