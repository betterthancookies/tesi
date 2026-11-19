from __future__ import annotations

from device.constants import (
    BATCH_OVERHEAD_S,
    MODEL_MACS_FWD_PER_SAMPLE,
    TRAIN_MACS_FACTOR,
)
from device.device_profile import DeviceProfile


def training_macs(epochs: int, n_samples: int) -> float:
    return TRAIN_MACS_FACTOR * MODEL_MACS_FWD_PER_SAMPLE * epochs * n_samples


def training_time_s(
    profile: DeviceProfile, epochs: int, n_samples: int, batch_size: int
) -> float:
    n_batches = max(1, n_samples // batch_size)
    compute_s = training_macs(epochs, n_samples) / profile.macs_per_s
    overhead_s = epochs * n_batches * BATCH_OVERHEAD_S
    return compute_s + overhead_s


def probe_time_s(profile: DeviceProfile, n_samples: int) -> float:
    if n_samples <= 0:
        return 0.0
    return MODEL_MACS_FWD_PER_SAMPLE * n_samples / profile.macs_per_s


if __name__ == "__main__":
    from device.device_profile import generate_profiles

    for p in generate_profiles(3, seed=0):
        print(f"{p.tier:>4}: {p.macs_per_s / 1e9:.1f} GMAC/s, {p.train_power_w:.2f} W")
        for b in (32, 64, 128, 256):
            t = training_time_s(p, 1, 1666, b)
            print(f"   B={b:3d}: epoca da 1666 campioni {t:5.1f} s, "
                  f"{p.train_power_w * t / 1666 * 1000:5.1f} mJ/campione")
