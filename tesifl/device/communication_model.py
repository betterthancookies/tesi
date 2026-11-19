from __future__ import annotations

from device.constants import (
    COMM_BANDWIDTH_DL_MB_PER_S,
    COMM_BANDWIDTH_UL_MB_PER_S,
    COMM_ENERGY_DL_J_PER_MB,
    COMM_ENERGY_UL_J_PER_MB,
    COMM_RTT_S,
    CTRL_PAYLOAD_MB,
    MODEL_SIZE_MB,
    SESSION_PROMO_POWER_W,
    SESSION_PROMO_TIME_S,
    SESSION_TAIL_POWER_W,
    SESSION_TAIL_TIME_S,
)
from device.device_profile import DeviceProfile


def round_payload_mb() -> tuple[float, float]:
    return MODEL_SIZE_MB, MODEL_SIZE_MB


def _payload_energy_j(rx_mb: float, tx_mb: float) -> float:
    return rx_mb * COMM_ENERGY_DL_J_PER_MB + tx_mb * COMM_ENERGY_UL_J_PER_MB


def _payload_time_s(rx_mb: float, tx_mb: float) -> float:
    return rx_mb / COMM_BANDWIDTH_DL_MB_PER_S + tx_mb / COMM_BANDWIDTH_UL_MB_PER_S


def session_energy_j() -> float:
    return (SESSION_PROMO_POWER_W * SESSION_PROMO_TIME_S
            + SESSION_TAIL_POWER_W * SESSION_TAIL_TIME_S)


def session_time_s() -> float:
    return SESSION_PROMO_TIME_S + SESSION_TAIL_TIME_S


def comm_time_s(profile: DeviceProfile | None = None) -> float:
    rx_mb, tx_mb = round_payload_mb()
    return _payload_time_s(rx_mb, tx_mb) + 2 * session_time_s()


def comm_energy_j(profile: DeviceProfile | None = None) -> float:
    rx_mb, tx_mb = round_payload_mb()
    return _payload_energy_j(rx_mb, tx_mb) + 2 * session_energy_j()


def comm_power_w(profile: DeviceProfile | None = None) -> float:
    t = comm_time_s()
    return comm_energy_j() / t if t > 0 else 0.0


def download_time_s() -> float:
    rx_mb, _ = round_payload_mb()
    return _payload_time_s(rx_mb, 0.0) + session_time_s()


def download_energy_j() -> float:
    rx_mb, _ = round_payload_mb()
    return _payload_energy_j(rx_mb, 0.0) + session_energy_j()


def download_power_w() -> float:
    t = download_time_s()
    return download_energy_j() / t if t > 0 else 0.0


def ctrl_energy_j(merged: bool = False) -> float:
    half = CTRL_PAYLOAD_MB / 2.0
    return _payload_energy_j(half, half) + (0.0 if merged else session_energy_j())


def ctrl_time_s(merged: bool = False) -> float:
    half = CTRL_PAYLOAD_MB / 2.0
    return _payload_time_s(half, half) + (0.0 if merged else session_time_s())


def ctrl_power_w(merged: bool = False) -> float:
    t = ctrl_time_s(merged)
    return ctrl_energy_j(merged) / t if t > 0 else 0.0


def ctrl_latency_s() -> float:
    half = CTRL_PAYLOAD_MB / 2.0
    return SESSION_PROMO_TIME_S + COMM_RTT_S + _payload_time_s(half, half)


if __name__ == "__main__":
    rx, tx = round_payload_mb()
    print(f"payload per round: RX={rx:.3f} MB, TX={tx:.3f} MB (simmetrico)")
    print(f"payload: {_payload_energy_j(rx, tx):.2f} J in {_payload_time_s(rx, tx):.2f} s")
    print(f"sessione radio: {session_energy_j():.2f} J in {session_time_s():.2f} s")
    print(f"modello: t={comm_time_s():.2f}s  E={comm_energy_j():.2f}J  "
          f"P_media={comm_power_w():.2f}W")
    for m in (False, True):
        print(f"controllo (merged={m}): E={ctrl_energy_j(m):.4f}J "
              f"t={ctrl_time_s(m):.3f}s  latenza round {ctrl_latency_s():.3f}s")
