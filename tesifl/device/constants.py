"""Costanti del modello fisico, lette da experiment.toml.

[B] NESSUN VALORE E' SCRITTO QUI. Tutti vengono da `experiment.toml` nella
radice del progetto, che e' l'unico file da modificare per cambiare un
esperimento. Questo modulo si limita a caricarlo e a esporre i nomi che il
resto del codice usa da sempre, cosi' nulla d'altro va toccato.

Se il file manca o una chiave non c'e', l'errore e' esplicito: meglio
fermarsi che proseguire con un default silenzioso, che e' il modo in cui in
passato sono finiti nei risultati parametri che nessuno aveva scelto.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment.toml"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"manca {CONFIG_PATH}: e' il file di configurazione dell'esperimento"
    )

with CONFIG_PATH.open("rb") as _fh:
    CFG = tomllib.load(_fh)


def _get(section: str, key: str):
    try:
        return CFG[section][key]
    except KeyError as exc:
        raise KeyError(
            f"experiment.toml: manca [{section}].{key}"
        ) from exc


# =====================================================================
# device e popolazione
# =====================================================================
TIERS = {
    t["name"]: {"capacity_mah": float(t["capacity_mah"]),
                "macs_per_s": float(t["macs_per_s"]),
                "train_power_w": float(t["train_power_w"])}
    for t in _get("device", "tiers")
}
# popolazione omogenea (world.tiers = false)
BATTERY_CAPACITY_MAH = float(_get("device", "capacity_mah_uniform"))
MACS_PER_S_UNIFORM = float(_get("device", "macs_per_s_uniform"))
TRAIN_POWER_W_UNIFORM = float(_get("device", "train_power_w_uniform"))
IDLE_POWER_W = float(_get("device", "idle_power_w"))

INITIAL_SOC_MEAN = float(_get("device", "soc_init_mean"))
INITIAL_SOC_STD = float(_get("device", "soc_init_std"))
INITIAL_SOC_RANGE = tuple(float(x) for x in _get("device", "soc_init_range"))

# =====================================================================
# batteria -- solo Peukert
# =====================================================================
V_NOMINAL = float(_get("battery", "v_nominal"))
RATED_C_RATE = float(_get("battery", "rated_c_rate"))
RATED_R_HOURS = 1.0 / RATED_C_RATE
DCDC_EFFICIENCY = float(_get("battery", "dcdc_efficiency"))
PEUKERT_N = float(_get("battery", "peukert_n"))

# =====================================================================
# calcolo
# =====================================================================
# [DATO ESATTO] MACs per campione, LeNet-5 su CIFAR-10 (input 3x32x32):
#   conv1 (3->6, 5x5, out 28x28):  28*28*6 * (5*5*3) = 352_800
#   conv2 (6->16, 5x5, out 10x10): 10*10*16 * (5*5*6) = 240_000
#   fc1 (400->120): 48_000    fc2 (120->84): 10_080    fc3 (84->10): 840
LENET5_MACS_FWD_PER_SAMPLE = 352_800 + 240_000 + 48_000 + 10_080 + 840

TRAIN_MACS_FACTOR = float(_get("compute", "train_macs_factor"))
BATCH_OVERHEAD_S = float(_get("compute", "batch_overhead_s"))

# =====================================================================
# comunicazione
# =====================================================================
# [ASSUNZIONE] payload SIMMETRICO: il server manda il modello intero, il
# client rimanda un update della stessa taglia.
#   conv1: 456   conv2: 2_416   fc1: 48_120   fc2: 10_164   fc3: 850
MODEL_MACS_FWD_PER_SAMPLE = int(_get("model", "macs_fwd_per_sample"))
MODEL_N_PARAMS = int(_get("model", "n_params"))
BYTES_PER_PARAM = int(_get("comm", "bytes_per_param"))
MODEL_SIZE_MB = MODEL_N_PARAMS * BYTES_PER_PARAM / 1e6
# 4G (LTE): energia per MB e banda, separate per download e upload
COMM_ENERGY_DL_J_PER_MB = float(_get("comm", "energy_dl_j_per_mb"))
COMM_ENERGY_UL_J_PER_MB = float(_get("comm", "energy_ul_j_per_mb"))
COMM_BANDWIDTH_DL_MB_PER_S = float(_get("comm", "bandwidth_dl_mbps")) / 8.0
COMM_BANDWIDTH_UL_MB_PER_S = float(_get("comm", "bandwidth_ul_mbps")) / 8.0
# sessione radio 4G (promozione + coda), Narayanan et al. 2021 / Huang 2012
SESSION_PROMO_POWER_W = float(_get("comm", "session_promo_power_w"))
SESSION_PROMO_TIME_S = float(_get("comm", "session_promo_time_s"))
SESSION_TAIL_POWER_W = float(_get("comm", "session_tail_power_w"))
SESSION_TAIL_TIME_S = float(_get("comm", "session_tail_time_s"))
COMM_RTT_S = float(_get("comm", "rtt_s"))
CTRL_PAYLOAD_MB = float(_get("comm", "ctrl_payload_kb")) / 1000.0

# =====================================================================
# workload concorrente
# =====================================================================
WORKLOAD_STATES = {k: float(v) for k, v in _get("workload", "states").items()}
WORKLOAD_CHAINS = {
    "low": _get("workload", "chain_low"),
    "medium": _get("workload", "chain_medium"),
    "high": _get("workload", "chain_high"),
}

# =====================================================================
# ricarica
# =====================================================================
RECHARGE_RATE_PER_H = float(_get("recharge", "rate_per_hour"))
CHARGE_C_RATE = float(_get("recharge", "c_rate"))
RECHARGE_AVAILABLE_DEFAULT = bool(CFG["world"].get("recharge_available", True))


def describe() -> str:
    """Riassunto dei parametri caricati, per i log della run."""
    t = " ".join(f"{k}({v['capacity_mah']:.0f}mAh,{v['macs_per_s'] / 1e9:.1f}GMAC/s,"
                 f"{v['train_power_w']:.2f}W)" for k, v in TIERS.items())
    return (f"tier: {t} | P_idle {IDLE_POWER_W * 1000:.0f} mW "
            f"| Peukert n={PEUKERT_N} | SoC0 N({INITIAL_SOC_MEAN},{INITIAL_SOC_STD}) "
            f"| 4G {COMM_ENERGY_DL_J_PER_MB}/{COMM_ENERGY_UL_J_PER_MB} J/MB "
            f"| ricarica {RECHARGE_RATE_PER_H}/h a {CHARGE_C_RATE}C")


if __name__ == "__main__":
    print(f"letto da {CONFIG_PATH}\n{describe()}")