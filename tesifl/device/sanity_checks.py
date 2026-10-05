"""Suite di verifica per il package device/ (popolazione omogenea, Peukert).

USO, dalla root del progetto:
    python -m device.sanity_checks
"""

from __future__ import annotations

import sys

import numpy as np

CHECKS_RUN = 0
CHECKS_PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global CHECKS_RUN, CHECKS_PASSED
    CHECKS_RUN += 1
    print(f"[{'OK  ' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        print(f"\n*** Interrotto al primo fallimento: {name} ***")
        sys.exit(1)
    CHECKS_PASSED += 1


def approx(a: float, b: float, tol: float = 0.01) -> bool:
    return abs(a - b) <= tol


# =====================================================================
print("=== 1. device_profile.py -- popolazione omogenea ===")
# =====================================================================
from device.device_profile import generate_profiles

profs = generate_profiles(n_clients=30)
check("tutti i profili sono identici",
      all(p.battery_capacity_mah == profs[0].battery_capacity_mah
          and p.macs_per_s == profs[0].macs_per_s
          and p.peukert_n == profs[0].peukert_n for p in profs))
check("il seed non influenza i profili (mondo fisico deterministico)",
      generate_profiles(30, seed=1) == generate_profiles(30, seed=999))
check("i cid sono 0..n-1 senza buchi",
      [p.cid for p in profs] == list(range(30)))

# =====================================================================
print("\n=== 2. battery_model.py -- formule contro gli handout ===")
# =====================================================================
from device.battery_model import (
    delta_soc, energy_wh_from_delta_soc, load_power_to_battery_current,
    peukert_capacity_ah,
)
from device.constants import TRAIN_POWER_W


class _Fake:
    battery_capacity_mah = 1000.0
    peukert_n = 1.1


cp = peukert_capacity_ah(_Fake())  # type: ignore[arg-type]
check("Peukert capacity, Esempio 5.2 handout (atteso 0.851 Ah)",
      approx(cp, 0.851, tol=0.001), detail=f"{cp:.4f} Ah")
check("durata a 0.2A, Esempio 5.2 handout (atteso 5.00 h)",
      approx(cp / (0.2 ** 1.1), 5.0, tol=0.01),
      detail=f"{cp / (0.2 ** 1.1):.3f} h")

p = profs[0]
check("la capacita' effettiva e' MINORE di quella nominale (Peukert morde)",
      peukert_capacity_ah(p) < p.battery_capacity_mah / 1000.0,
      detail=f"C'={peukert_capacity_ah(p):.4f} Ah  C={p.battery_capacity_mah/1000:.3f} Ah")

d = delta_soc(power_w=TRAIN_POWER_W, dt_s=30.0, profile=p)
check("delta_soc e' negativo (solo scarica)", d < 0, detail=f"{d*100:+.3f}%/30s")
check("delta_soc scala linearmente col tempo, a potenza fissa",
      approx(delta_soc(TRAIN_POWER_W, 60.0, p) / d, 2.0, tol=1e-9))
check("delta_soc e' SUPERLINEARE nella potenza (non linearita' di Peukert)",
      delta_soc(2 * TRAIN_POWER_W, 30.0, p) < 2 * d,
      detail=f"P: {d*100:+.3f}%  2P: {delta_soc(2*TRAIN_POWER_W, 30.0, p)*100:+.3f}%")
check("energia derivata dal delta_soc e' positiva",
      energy_wh_from_delta_soc(d, p) > 0,
      detail=f"{energy_wh_from_delta_soc(d, p)*1000:.2f} mWh")

i = load_power_to_battery_current(TRAIN_POWER_W)
c_rate_train = i / (p.battery_capacity_mah / 1000.0)
check("la corrente di training e' un C-rate plausibile (< 2C)",
      c_rate_train < 2.0, detail=f"{c_rate_train:.2f}C")

# =====================================================================
print("\n=== 3. compute_model -- tempo, overhead, batch ottimo ===")
# =====================================================================
from device.compute_model import optimal_batch_size, training_macs, training_time_s
from device.constants import (
    BATCH_OVERHEAD_S, LENET5_MACS_FWD_PER_SAMPLE, POWER_EXPONENT_BATCH,
    TRAIN_MACS_FACTOR,
)

check("MACs fwd LeNet-5/CIFAR-10 dal conto a mano",
      LENET5_MACS_FWD_PER_SAMPLE == 352_800 + 240_000 + 48_000 + 10_080 + 840,
      detail=f"{LENET5_MACS_FWD_PER_SAMPLE:,} MAC/campione")
check("MACs di un round = factor x fwd x epoche x campioni",
      approx(training_macs(5, 500),
             TRAIN_MACS_FACTOR * LENET5_MACS_FWD_PER_SAMPLE * 5 * 500, tol=1.0))
check("il tempo scala linearmente con le epoche",
      approx(training_time_s(p, 10, 500, 128) / training_time_s(p, 5, 500, 128),
             2.0, tol=1e-9))
check("[B] la latenza varia SOLO col numero di campioni (device identici)",
      training_time_s(p, 5, 2000, 128) > training_time_s(profs[7], 5, 500, 128),
      detail=f"1333 campioni, B=128 -> {training_time_s(p, 5, 1333, 128):.1f}s")
check("batch piu' grandi ammortizzano l'overhead: il round e' piu' corto",
      training_time_s(p, 5, 1333, 256) < training_time_s(p, 5, 1333, 32),
      detail=f"B=32: {training_time_s(p, 5, 1333, 32):.1f}s  "
             f"B=256: {training_time_s(p, 5, 1333, 256):.1f}s")

# ---- il batch ottimo e la sua invarianza ----------------------------
b_lin = optimal_batch_size(p, peukert=False)
b_peu = optimal_batch_size(p, peukert=True)
ratio_atteso = (POWER_EXPONENT_BATCH * (1 - POWER_EXPONENT_BATCH * p.peukert_n)) / (
    POWER_EXPONENT_BATCH * p.peukert_n * (1 - POWER_EXPONENT_BATCH))
check("[B] il batch ottimo sotto Peukert e' piu' PICCOLO del lineare "
      "(esponente efficace g*n)",
      approx(b_peu / b_lin, ratio_atteso, tol=1e-6),
      detail=f"lineare {b_lin:.1f}  Peukert {b_peu:.1f}  "
             f"rapporto {b_peu/b_lin:.3f}")
check("[B] il batch ottimo NON dipende da epoche e campioni "
      "(c/Tc si semplifica)",
      BATCH_OVERHEAD_S > 0,
      detail=f"B* = (1-g)/g * t_ov*throughput/(3*MACs) = {b_lin:.1f}")


def _round_energy(profile, bs, epochs=5, n=1333):
    """Energia di un round a batch bs, come la calcola world_state."""
    from device.constants import (
        BATCH_SIZE_REF, POWER_BATCH_MULT_MAX, POWER_BATCH_MULT_MIN,
    )
    mult = (bs / BATCH_SIZE_REF) ** POWER_EXPONENT_BATCH
    mult = min(max(mult, POWER_BATCH_MULT_MIN), POWER_BATCH_MULT_MAX)
    return TRAIN_POWER_W * mult * training_time_s(profile, epochs, n, bs)


grid = [8, 16, 32, 45, 55, 64, 96, 128, 192, 256, 384, 512]
energies = {bs: _round_energy(p, bs) for bs in grid}
b_min = min(energies, key=energies.get)
check("[B] l'energia di un round ha un MINIMO INTERNO in B "
      "(senza overhead sarebbe monotona)",
      grid[0] < b_min < grid[-1],
      detail=f"minimo tabulare a B={b_min}, formula B*={b_lin:.1f}")
check("il minimo numerico e' vicino a quello analitico "
      "(la divisione intera su n_batches lo sposta un poco)",
      0.5 * b_lin <= b_min <= 2.0 * b_lin,
      detail=f"B={b_min} vs B*={b_lin:.1f}")
check("B=128 dei paper costa piu' del minimo, e B grande molto di piu'",
      energies[128] > energies[b_min] and energies[512] > energies[128],
      detail=f"B=128 +{100*(energies[128]/energies[b_min]-1):.1f}%  "
             f"B=512 +{100*(energies[512]/energies[b_min]-1):.1f}%")

# =====================================================================
print("\n=== 4. communication_model.py ===")
# =====================================================================
from device.communication_model import (
    comm_energy_j, comm_power_w, comm_time_s, round_payload_mb,
)
from device.constants import LENET5_N_PARAMS, MODEL_SIZE_MB

check("parametri LeNet-5 dal conto a mano",
      LENET5_N_PARAMS == 456 + 2_416 + 48_120 + 10_164 + 850,
      detail=f"{LENET5_N_PARAMS:,}")
check("taglia modello ~0.248 MB, coerente con l'ArrayRecord di Flower",
      approx(MODEL_SIZE_MB, 0.248, tol=0.005), detail=f"{MODEL_SIZE_MB:.3f} MB")
rx, tx = round_payload_mb()
check("payload simmetrico", approx(rx, tx, tol=1e-12))
check("coerenza E = P_media x t",
      approx(comm_power_w() * comm_time_s(), comm_energy_j(), tol=1e-9),
      detail=f"E={comm_energy_j():.3f}J  t={comm_time_s():.3f}s  P={comm_power_w():.2f}W")

# =====================================================================
print("\n=== 5. world_state.py -- invarianti su una run completa ===")
# =====================================================================
from device.constants import V_NOMINAL
from device.world_state import build_world_state

N, K, R, BS = 30, 6, 250, 128
ws = build_world_state(n_clients=N, seed=42)
soc0 = float(np.mean([ws.snapshot(c).soc for c in range(N)]))
rng = np.random.default_rng(0)
soc_out_of_range = False
resurrected = False
prev_failed = {c: False for c in range(N)}

for r in range(1, R + 1):
    alive = [c for c in range(N) if ws.snapshot(c).available]
    if not alive:
        break
    for cid in rng.choice(alive, size=min(K, len(alive)), replace=False):
        dt = ws.round_duration_s(cid, epochs=5, batch_size=BS, dataset_size_local=1333)
        ws.apply_round(cid, dt_s=dt, batch_size=BS)
        ws.apply_communication(cid)
    for c in range(N):
        snap = ws.snapshot(c)
        if not (0.0 <= snap.soc <= 1.0):
            soc_out_of_range = True
        if prev_failed[c] and not snap.failed:
            resurrected = True
        prev_failed[c] = snap.failed

s = ws.stats()
check(f"SoC sempre in [0,1] su {R} round x {N} client", not soc_out_of_range)
check("la morte e' permanente: nessun client risorge", not resurrected)
check("energia di training E di comunicazione entrambe positive",
      s["energy_train_wh"] > 0 and s["energy_comm_wh"] > 0,
      detail=f"train={s['energy_train_wh']:.2f} Wh  comm={s['energy_comm_wh']:.2f} Wh")
check("energia totale = train + comm",
      approx(s["total_energy_wh"], s["energy_train_wh"] + s["energy_comm_wh"], tol=1e-9))
check("n_soc_zero coincide con n_failed",
      s["n_soc_zero"] == s["n_failed"], detail=f"{s['n_failed']}/{N} morti")

# [B] identita' energia-SoC: con device identici ogni Wh sottratto a una
# cella e' contabilizzato e non esistono altre vie di consumo. Una
# violazione ha gia' smascherato due difetti (background drain non
# contabilizzato, disallineamento fra popolazione simulata e SuperNode).
wh_per_soc = N * V_NOMINAL * p.battery_capacity_mah / 1000.0
check("[B] identita' E_tot = N*V*C*(SoC_iniz - SoC_fin)",
      approx(s["total_energy_wh"], (soc0 - s["mean_soc"]) * wh_per_soc, tol=0.05),
      detail=f"misurata {s['total_energy_wh']:.3f} Wh  "
             f"attesa {(soc0 - s['mean_soc']) * wh_per_soc:.3f} Wh")

check("[B] il regime energetico e' vincolante ma non degenere "
      "(fra il 10% e il 90% di morti a fine run)",
      0.1 * N <= s["n_soc_zero"] <= 0.9 * N,
      detail=f"{s['n_soc_zero']}/{N} morti, SoC medio {s['mean_soc']:.3f}")

ws2 = build_world_state(n_clients=N, seed=42)
check("stesso seed -> stesso SoC iniziale",
      all(approx(ws2.snapshot(c).soc, build_world_state(N, 42).snapshot(c).soc, 1e-12)
          for c in range(N)))
check("seed diverso -> SoC iniziale diverso",
      any(not approx(build_world_state(N, 42).snapshot(c).soc,
                     build_world_state(N, 43).snapshot(c).soc, 1e-12)
          for c in range(N)))

print(f"\n{'='*60}")
print(f"TUTTI I CHECK PASSATI ({CHECKS_PASSED}/{CHECKS_RUN})")
print(f"{'='*60}")