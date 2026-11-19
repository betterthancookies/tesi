from __future__ import annotations

import numpy as np

from device.constants import DCDC_EFFICIENCY, V_NOMINAL
from device.device_profile import DeviceProfile

_SOC = np.array([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15,
                 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00])

_V_02C = np.array([2.50, 3.00, 3.15, 3.26, 3.33, 3.38, 3.44, 3.46, 3.51,
                   3.54, 3.57, 3.59, 3.61, 3.64, 3.66, 3.69, 3.71, 3.75,
                   3.80, 3.85, 3.90, 3.95, 4.03, 4.09, 4.17, 4.22, 4.24])
_V_05C = np.array([2.50, 2.80, 2.94, 3.04, 3.11, 3.16, 3.24, 3.29, 3.36,
                   3.41, 3.44, 3.47, 3.49, 3.51, 3.53, 3.55, 3.59, 3.62,
                   3.66, 3.71, 3.76, 3.81, 3.89, 3.96, 4.03, 4.08, 4.12])
_V_1C = np.array([2.50, 2.68, 2.78, 2.86, 2.91, 2.96, 3.04, 3.10, 3.18,
                  3.23, 3.27, 3.30, 3.33, 3.35, 3.38, 3.40, 3.43, 3.47,
                  3.50, 3.55, 3.59, 3.65, 3.72, 3.78, 3.84, 3.89, 3.95])

_CRATES = np.array([0.2, 0.5, 1.0])
_VTAB = np.vstack([_V_02C, _V_05C, _V_1C])

_QFRAC = np.array([2.60, 2.50, 2.39]) / 2.60

_BISECT_ITERS = 60


def _voltage(soc: float, crate: float) -> float:
    soc = float(np.clip(soc, 0.0, 1.0))
    v_at = np.array([np.interp(soc, _SOC, _VTAB[i]) for i in range(3)])
    if crate <= _CRATES[0]:
        return float(v_at[0])
    if crate >= _CRATES[-1]:
        slope = (v_at[2] - v_at[1]) / (_CRATES[2] - _CRATES[1])
        return float(max(0.5, v_at[2] + slope * (crate - _CRATES[-1])))
    return float(np.interp(crate, _CRATES, v_at))


def _q_fraction(crate: float) -> float:
    if crate <= _CRATES[0]:
        return float(_QFRAC[0])
    if crate >= _CRATES[-1]:
        slope = (_QFRAC[2] - _QFRAC[1]) / (_CRATES[2] - _CRATES[1])
        return float(max(0.3, _QFRAC[2] + slope * (crate - _CRATES[-1])))
    return float(np.interp(crate, _CRATES, _QFRAC))


def solve_current_a(power_w: float, soc: float, profile: DeviceProfile) -> float:
    if power_w <= 0.0:
        return 0.0
    q_nom_ah = profile.battery_capacity_mah / 1000.0
    p_batt = power_w / DCDC_EFFICIENCY
    lo, hi = 1e-9, 50.0 * q_nom_ah
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if _voltage(soc, mid / q_nom_ah) * mid < p_batt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def delta_soc(power_w: float, dt_s: float, profile: DeviceProfile,
              soc: float = 1.0) -> float:
    if power_w <= 0.0 or dt_s <= 0.0:
        return 0.0
    q_nom_ah = profile.battery_capacity_mah / 1000.0
    i_a = solve_current_a(power_w, soc, profile)
    q_eff_ah = q_nom_ah * _q_fraction(i_a / q_nom_ah)
    return -(i_a * (dt_s / 3600.0)) / q_eff_ah


def load_power_to_battery_current(power_w: float,
                                  profile: DeviceProfile | None = None,
                                  soc: float = 1.0) -> float:
    if power_w <= 0.0:
        return 0.0
    if profile is None:
        return (power_w / DCDC_EFFICIENCY) / V_NOMINAL
    return solve_current_a(power_w, soc, profile)


def energy_wh_linear(power_w: float, dt_s: float) -> float:
    if power_w <= 0.0 or dt_s <= 0.0:
        return 0.0
    return (power_w / DCDC_EFFICIENCY) * dt_s / 3600.0


def energy_wh_from_delta_soc(delta: float, profile: DeviceProfile) -> float:
    q_nom_ah = profile.battery_capacity_mah / 1000.0
    return V_NOMINAL * (-delta * q_nom_ah)


def c_rate(current_a: float, profile: DeviceProfile) -> float:
    return current_a / (profile.battery_capacity_mah / 1000.0)


def model_error_vs_linear(power_w: float, dt_s: float, profile: DeviceProfile,
                          soc: float) -> float:
    lin = energy_wh_linear(power_w, dt_s)
    if lin <= 0.0:
        return 1.0
    real = energy_wh_from_delta_soc(delta_soc(power_w, dt_s, profile, soc), profile)
    return real / lin


if __name__ == "__main__":
    from device.device_profile import generate_profiles

    p = generate_profiles(1)[0]
    TRAIN_POWER_W = p.train_power_w
    q = p.battery_capacity_mah / 1000.0
    print(f"cella {p.battery_capacity_mah:.0f} mAh, P_train {TRAIN_POWER_W} W ({p.tier})\n")
    print(f"{'SoC':>6}{'V':>8}{'I [A]':>8}{'C-rate':>8}{'Qeff/Qnom':>11}"
          f"{'dSoC/60s':>11}{'vs lineare':>12}")
    for s in (1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05):
        i = solve_current_a(TRAIN_POWER_W, s, p)
        cr = i / q
        d = delta_soc(TRAIN_POWER_W, 60.0, p, s)
        print(f"{s:6.2f}{_voltage(s, cr):8.3f}{i:8.3f}{cr:8.3f}"
              f"{_q_fraction(cr):11.3f}{d:11.5f}"
              f"{model_error_vs_linear(TRAIN_POWER_W, 60.0, p, s):11.3f}x")

    print("\n[verifica] esempio del documento: cella 2.6 Ah, P=5 W, SoC=50%")

    class _Cell:
        battery_capacity_mah = 2600.0
        peukert_n = 1.0
    i = solve_current_a(5.0, 0.50, _Cell())
    d = delta_soc(5.0, 60.0, _Cell(), 0.50)
    print(f"  I = {i:.3f} A ({i/2.6:.2f}C), V = {_voltage(0.5, i/2.6):.3f} V, "
          f"V*I = {_voltage(0.5, i/2.6)*i:.2f} W")
    print(f"  dSoC su 60 s = {d:+.5f}  (documento: -0.00816)")
