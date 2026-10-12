from __future__ import annotations

from device.constants import (
    BATCH_OVERHEAD_S,
    MODEL_MACS_FWD_PER_SAMPLE,
    TRAIN_MACS_FACTOR,
)
from device.device_profile import DeviceProfile


def training_macs(epochs: int, n_samples: int) -> float:
    #MACs totali di un round locale: fwd*factor x campioni x epoche
    return TRAIN_MACS_FACTOR * MODEL_MACS_FWD_PER_SAMPLE * epochs * n_samples


def training_time_s(
    profile: DeviceProfile, epochs: int, n_samples: int, batch_size: int
) -> float:
    """Durata di un round: calcolo + overhead per batch.

    [B] senza il secondo termine il batch size non influenzerebbe il TEMPO,
    solo la potenza, e il batch energeticamente ottimo sarebbe degeneremente
    il piu' piccolo disponibile. Con l'overhead l'energia ha un minimo
    interno, e B diventa una variabile decisionale vera.
    """
    n_batches = max(1, n_samples // batch_size)
    compute_s = training_macs(epochs, n_samples) / profile.macs_per_s
    overhead_s = epochs * n_batches * BATCH_OVERHEAD_S
    return compute_s + overhead_s


def probe_time_s(profile: DeviceProfile, n_samples: int) -> float:
    """Solo forward su n_samples: la stima della loss locale prima del training.

    [B] niente fattore 3 (non c'e' backward) e niente overhead per batch: e'
    una singola passata in valutazione, trascurabile rispetto al resto.
    """
    if n_samples <= 0:
        return 0.0
    return MODEL_MACS_FWD_PER_SAMPLE * n_samples / profile.macs_per_s


def optimal_batch_size(
    profile: DeviceProfile, peukert: bool = True, power_exponent: float = 0.3
) -> float:
    """Batch size che minimizza il costo di un round.

    Con P(B) = P0 (B/Bref)^g e t(B) = Tc + c/B, il costo e' minimo in

        B* = (1-g)/g * c/Tc,      c/Tc = t_ov * throughput / (3 * MACs_fwd)

    [B] il rapporto c/Tc NON dipende ne' dalle epoche ne' dai campioni: si
    semplificano. Il batch ottimo e' quindi una proprieta' del DEVICE e
    dell'architettura, calcolabile una volta sola dal client.

    [B] sotto Peukert il costo e' proporzionale a I^n * t, cioe' P^n * t,
    quindi l'esponente efficace diventa g*n e l'ottimo si sposta:
    con g=0.3 e n=1.15 il batch ottimo e' ~19% piu' piccolo di quello
    che minimizza l'energia lineare. Un client che sceglie B guardando i Wh
    sbaglia sistematicamente per eccesso.
    """
    g = power_exponent * (profile.peukert_n if peukert else 1.0)
    if not 0.0 < g < 1.0:
        raise ValueError(f"esponente efficace fuori range: {g}")
    c_over_tc = (BATCH_OVERHEAD_S * profile.macs_per_s
                 / (TRAIN_MACS_FACTOR * MODEL_MACS_FWD_PER_SAMPLE))
    return (1.0 - g) / g * c_over_tc


if __name__ == "__main__":
    from device.device_profile import generate_profiles

    p = generate_profiles(1)[0]
    print(f"B* lineare : {optimal_batch_size(p, peukert=False):.1f}")
    print(f"B* Peukert : {optimal_batch_size(p, peukert=True):.1f}")
    print(f"rapporto   : {optimal_batch_size(p) / optimal_batch_size(p, False):.3f}\n")

    for n in (500, 1333, 3000):
        t = training_time_s(p, 5, n, 128)
        print(f"{n:5d} campioni, E=5, B=128: {t:6.2f} s")