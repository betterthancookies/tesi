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


TIERS = {
    t["name"]: {"capacity_mah": float(t["capacity_mah"]),
                "macs_per_s": float(t["macs_per_s"]),
                "train_power_w": float(t["train_power_w"])}
    for t in _get("device", "tiers")
}
BATTERY_CAPACITY_MAH = float(_get("device", "capacity_mah_uniform"))
MACS_PER_S_UNIFORM = float(_get("device", "macs_per_s_uniform"))
TRAIN_POWER_W_UNIFORM = float(_get("device", "train_power_w_uniform"))
IDLE_POWER_W = float(_get("device", "idle_power_w"))
LOW_BATTERY_SOC = float(_get("device", "low_battery_soc"))

INITIAL_SOC_MEAN = float(_get("device", "soc_init_mean"))
INITIAL_SOC_STD = float(_get("device", "soc_init_std"))
INITIAL_SOC_RANGE = tuple(float(x) for x in _get("device", "soc_init_range"))

V_NOMINAL = float(_get("battery", "v_nominal"))
RATED_C_RATE = float(_get("battery", "rated_c_rate"))
RATED_R_HOURS = 1.0 / RATED_C_RATE
DCDC_EFFICIENCY = float(_get("battery", "dcdc_efficiency"))
PEUKERT_N = float(_get("battery", "peukert_n"))

LENET5_MACS_FWD_PER_SAMPLE = 352_800 + 240_000 + 48_000 + 10_080 + 840

TRAIN_MACS_FACTOR = float(_get("compute", "train_macs_factor"))
BATCH_OVERHEAD_S = float(_get("compute", "batch_overhead_s"))

MODEL_NAME = str(_get("model", "name"))
MODEL_DATASET = str(_get("model", "dataset"))
MODEL_AUGMENT = bool(CFG["model"].get("augment", False))
_SPEC_KEY = f"{MODEL_NAME}_{MODEL_DATASET}"
try:
    _SPEC = CFG["model"]["specs"][_SPEC_KEY]
except KeyError as exc:
    raise KeyError(f"experiment.toml: manca [model.specs.{_SPEC_KEY}] "
                   "(MAC e parametri di questo modello)") from exc
MODEL_MACS_FWD_PER_SAMPLE = int(_SPEC["macs_fwd_per_sample"])
MODEL_N_PARAMS = int(_SPEC["n_params"])
BYTES_PER_PARAM = int(_get("comm", "bytes_per_param"))
MODEL_SIZE_MB = MODEL_N_PARAMS * BYTES_PER_PARAM / 1e6
COMM_ENERGY_DL_J_PER_MB = float(_get("comm", "energy_dl_j_per_mb"))
COMM_ENERGY_UL_J_PER_MB = float(_get("comm", "energy_ul_j_per_mb"))
COMM_BANDWIDTH_DL_MB_PER_S = float(_get("comm", "bandwidth_dl_mbps")) / 8.0
COMM_BANDWIDTH_UL_MB_PER_S = float(_get("comm", "bandwidth_ul_mbps")) / 8.0
SESSION_PROMO_POWER_W = float(_get("comm", "session_promo_power_w"))
SESSION_PROMO_TIME_S = float(_get("comm", "session_promo_time_s"))
SESSION_TAIL_POWER_W = float(_get("comm", "session_tail_power_w"))
SESSION_TAIL_TIME_S = float(_get("comm", "session_tail_time_s"))
COMM_RTT_S = float(_get("comm", "rtt_s"))
CTRL_PAYLOAD_MB = float(_get("comm", "ctrl_payload_kb")) / 1000.0

WORKLOAD_STATES = {k: float(v) for k, v in _get("workload", "states").items()}
WORKLOAD_CHAINS = {
    "low": _get("workload", "chain_low"),
    "medium": _get("workload", "chain_medium"),
    "high": _get("workload", "chain_high"),
}

RECHARGE_RATE_PER_H = float(_get("recharge", "rate_per_hour"))
CHARGE_C_RATE = float(_get("recharge", "c_rate"))
RECHARGE_AVAILABLE_DEFAULT = bool(CFG["world"].get("recharge_available", True))


def describe() -> str:
    t = " ".join(f"{k}({v['capacity_mah']:.0f}mAh,{v['macs_per_s'] / 1e9:.1f}GMAC/s,"
                 f"{v['train_power_w']:.2f}W)" for k, v in TIERS.items())
    return (f"modello {MODEL_NAME}/{MODEL_DATASET} ({MODEL_MACS_FWD_PER_SAMPLE / 1e6:.1f} MMAC, "
            f"{MODEL_SIZE_MB:.1f} MB, augment {'on' if MODEL_AUGMENT else 'off'}) "
            f"| tier: {t} | P_idle {IDLE_POWER_W * 1000:.0f} mW "
            f"| Peukert n={PEUKERT_N} | SoC0 N({INITIAL_SOC_MEAN},{INITIAL_SOC_STD}) "
            f"| 4G {COMM_ENERGY_DL_J_PER_MB}/{COMM_ENERGY_UL_J_PER_MB} J/MB "
            f"| ricarica {RECHARGE_RATE_PER_H}/h a {CHARGE_C_RATE}C")


if __name__ == "__main__":
    print(f"letto da {CONFIG_PATH}\n{describe()}")
