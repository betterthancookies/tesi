"""Costo energetico e temporale della comunicazione.

Due componenti, entrambe per device e per round:
  - payload: J/MB in ricezione e trasmissione (energy_rx_j_per_mb, ratio TX);
  - SESSIONE radio: ogni trasferimento isolato attiva la radio LTE, che paga
    una promozione IDLE->CONNECTED e poi resta accesa per Ttail prima di
    tornare IDLE (Huang et al., MobiSys 2012, Tabella 3). Costo fisso, ~12.6 J,
    indipendente dalla taglia del messaggio.

[B] PERCHE' LA SESSIONE CONTA. Per il modello (~1 MB) il payload costa ~4 J e
la sessione ~25 J (download e upload sono due sessioni: in mezzo c'e' il
training, quasi sempre piu' lungo di Ttail). Per un messaggio di controllo da
1 KB il payload costa ~1e-3 J e la sessione tutto il resto: senza sessione,
interrogare lo stato dei client sembrerebbe gratis.

[B] LA CODA NON ALLUNGA IL ROUND: la radio resta accesa dopo il trasferimento,
mentre il device fa altro. Entra nell'energia (e nella potenza media vista
dalla batteria), non nella durata del round. Il round si allunga solo della
latenza di un messaggio di controllo: promozione + RTT + trasmissione.

[ASSUNZIONE] durante la coda il consumo di base dell'idle viene contato due
volte (0.1 W contro 1.06 W della coda): errore trascurabile e conservativo.
"""
from __future__ import annotations

from device.constants import (
    COMM_BANDWIDTH_MB_PER_S,
    COMM_ENERGY_RX_J_PER_MB,
    COMM_RTT_S,
    COMM_TX_RX_ENERGY_RATIO,
    CTRL_PAYLOAD_MB,
    MODEL_SIZE_MB,
    SESSION_PROMO_POWER_W,
    SESSION_PROMO_TIME_S,
    SESSION_TAIL_POWER_W,
    SESSION_TAIL_TIME_S,
)
from device.device_profile import DeviceProfile


def round_payload_mb() -> tuple[float, float]:
    #(rx_mb, tx_mb) di un round: modello intero in giu', update stessa taglia in su
    return MODEL_SIZE_MB, MODEL_SIZE_MB


def _payload_energy_j(rx_mb: float, tx_mb: float) -> float:
    return (rx_mb * COMM_ENERGY_RX_J_PER_MB
            + tx_mb * COMM_ENERGY_RX_J_PER_MB * COMM_TX_RX_ENERGY_RATIO)


def session_energy_j() -> float:
    """Energia fissa di una sessione radio: promozione + coda."""
    return (SESSION_PROMO_POWER_W * SESSION_PROMO_TIME_S
            + SESSION_TAIL_POWER_W * SESSION_TAIL_TIME_S)


def session_time_s() -> float:
    return SESSION_PROMO_TIME_S + SESSION_TAIL_TIME_S


# --------------------------------------------------- scambio del modello
def comm_time_s(profile: DeviceProfile | None = None) -> float:
    #RX e TX sequenziali sulla stessa banda (non sovrapposti) [ASSUNZIONE], piu'
    #le due sessioni radio. `profile` accettato e ignorato: rete unica.
    rx_mb, tx_mb = round_payload_mb()
    return (rx_mb + tx_mb) / COMM_BANDWIDTH_MB_PER_S + 2 * session_time_s()


def comm_energy_j(profile: DeviceProfile | None = None) -> float:
    rx_mb, tx_mb = round_payload_mb()
    return _payload_energy_j(rx_mb, tx_mb) + 2 * session_energy_j()


def comm_power_w(profile: DeviceProfile | None = None) -> float:
    #potenza media equivalente: serve al modello di batteria, che ragiona in
    #potenza x tempo, cosi' la non-idealita' si applica anche alla comunicazione
    t = comm_time_s()
    return comm_energy_j() / t if t > 0 else 0.0


# ------------------------------------------------- messaggi di controllo
# Richiesta di stato o proposta di lavoro, con la risposta del client.
# merged=True: lo scambio cade nella stessa sessione radio del download del
# modello che segue subito dopo (client selezionato), quindi non paga una
# sessione propria. merged=False: sessione isolata, costo pieno.
def ctrl_energy_j(merged: bool = False) -> float:
    half = CTRL_PAYLOAD_MB / 2.0
    return _payload_energy_j(half, half) + (0.0 if merged else session_energy_j())


def ctrl_time_s(merged: bool = False) -> float:
    """Durata della scarica: trasmissione piu', se isolata, la sessione."""
    t = CTRL_PAYLOAD_MB / COMM_BANDWIDTH_MB_PER_S
    return t + (0.0 if merged else session_time_s())


def ctrl_power_w(merged: bool = False) -> float:
    t = ctrl_time_s(merged)
    return ctrl_energy_j(merged) / t if t > 0 else 0.0


def ctrl_latency_s() -> float:
    """Quanto uno scambio di controllo ritarda il round: promozione + RTT +
    trasmissione. La coda non conta, la radio si spegne in background."""
    return SESSION_PROMO_TIME_S + COMM_RTT_S + CTRL_PAYLOAD_MB / COMM_BANDWIDTH_MB_PER_S


if __name__ == "__main__":
    rx, tx = round_payload_mb()
    print(f"payload per round: RX={rx:.3f} MB, TX={tx:.3f} MB (simmetrico)")
    print(f"sessione radio: {session_energy_j():.2f} J in {session_time_s():.2f} s")
    print(f"modello: t={comm_time_s():.2f}s  E={comm_energy_j():.2f}J  "
          f"P_media={comm_power_w():.2f}W")
    for m in (False, True):
        print(f"controllo (merged={m}): E={ctrl_energy_j(m):.4f}J "
              f"t={ctrl_time_s(m):.3f}s  latenza round {ctrl_latency_s():.3f}s")
