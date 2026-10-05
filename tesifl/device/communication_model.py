from __future__ import annotations

from device.constants import (
    COMM_BANDWIDTH_MB_PER_S,
    COMM_ENERGY_RX_J_PER_MB,
    COMM_TX_RX_ENERGY_RATIO,
    MODEL_SIZE_MB,
)
from device.device_profile import DeviceProfile


def round_payload_mb() -> tuple[float, float]:
    #(rx_mb, tx_mb) di un round: modello intero in giu', update stessa taglia in su
    return MODEL_SIZE_MB, MODEL_SIZE_MB


def comm_time_s(profile: DeviceProfile | None = None) -> float:
    #RX e TX sequenziali sulla stessa banda (non sovrapposti) [ASSUNZIONE].
    #`profile` accettato e ignorato: rete unica per tutti i device.
    rx_mb, tx_mb = round_payload_mb()
    return (rx_mb + tx_mb) / COMM_BANDWIDTH_MB_PER_S


def comm_energy_j(profile: DeviceProfile | None = None) -> float:
    rx_mb, tx_mb = round_payload_mb()
    return (rx_mb * COMM_ENERGY_RX_J_PER_MB
            + tx_mb * COMM_ENERGY_RX_J_PER_MB * COMM_TX_RX_ENERGY_RATIO)


def comm_power_w(profile: DeviceProfile | None = None) -> float:
    #potenza media equivalente: serve al modello di batteria, che ragiona in
    #potenza x tempo, cosi' Peukert si applica anche alla comunicazione
    t = comm_time_s()
    return comm_energy_j() / t if t > 0 else 0.0


if __name__ == "__main__":
    rx, tx = round_payload_mb()
    print(f"payload per round: RX={rx:.3f} MB, TX={tx:.3f} MB (simmetrico)")
    print(f"t={comm_time_s():.3f}s  E={comm_energy_j():.3f}J  "
          f"P_media={comm_power_w():.2f}W")