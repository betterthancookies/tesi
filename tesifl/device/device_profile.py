"""Popolazione di client: tre tier hardware in parti uguali.

Ogni tier differisce per tre grandezze, capacita' di batteria, velocita' di
calcolo e potenza di training, prese da constants.TIERS (i tre telefoni di
Mandheling, vedi experiment.toml). Con `tiers_enabled=False` la popolazione
torna omogenea e identica a prima: e' l'ablazione che separa l'effetto
dell'eterogeneita' hardware da quello della politica di selezione.

[B] L'ASSEGNAZIONE E' PERMUTATA, non `cid % 3`. La taglia della partizione
Dirichlet dipende dal cid, quindi un'assegnazione ordinata correlerebbe il
tier con la quantita' di dati locali e i due effetti sarebbero inseparabili.
La permutazione dipende dal seed, quindi resta riproducibile.
"""

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
    # potenza del telefono durante il training, costante (non dipende da B)
    train_power_w: float = TRAIN_POWER_W_UNIFORM


def generate_profiles(
    n_clients: int,
    seed: int | None = None,
    capacity_mah: float | None = None,
    tiers_enabled: bool = True,
) -> list[DeviceProfile]:
    """Popolazione di n_clients device.

    tiers_enabled=True  tre tier in parti uguali, permutati col seed.
                        `capacity_mah` viene IGNORATA: la capacita' la
                        definisce il tier.
    tiers_enabled=False tutti identici, con `capacity_mah` o la costante.
                        E' la configurazione delle campagne precedenti.
    """
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
            # [B] la velocita' scala il throughput, quindi entra sia nel
            # consumo (round piu' corto = meno energia) sia nella durata del
            # round, che il modello di idle definisce come il training del
            # selezionato piu' lento: un tier lento penalizza anche gli altri.
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