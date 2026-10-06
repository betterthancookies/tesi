"""Modello di batteria di Peukert: il modello "peuk" del confronto.

    C' = C * (C/R)^(n-1)
    t_pieno = C' / I^n
    dSoC = -dt / t_pieno

applicato a tratti a potenza costante (un round = una potenza).

RUOLO NEL SIMULATORE. La batteria VERA del mondo segue il modello da
datasheet (battery_model.py, il modello "nm"). Peukert e' il modello che usa
il FUEL GAUGE dei bracci "peuk": WorldState.soc_peukert tiene, per ogni
device, il SoC che il device crederebbe di avere stimandolo con Peukert, e i
bracci sage_peuk, sage_smart_peuk ed escs_*_peuk selezionano (e il client di
sage_smart decide) su quel SoC. Come il braccio lineare, misura quanto conta
l'accuratezza del modello di batteria: lineare < Peukert < datasheet.

[B] esponente n = PEUKERT_N di experiment.toml (1.15), capacita' nominale
alla corrente di riferimento R = RATED_R_HOURS (rated_c_rate = 0.2C, 5 h).
Funzioni identiche al modello Peukert usato nelle campagne precedenti.
"""

from __future__ import annotations

from device.constants import DCDC_EFFICIENCY, RATED_R_HOURS, V_NOMINAL
from device.device_profile import DeviceProfile


def load_power_to_battery_current(power_w: float) -> float:
    """P_batt = P_load / eta_dcdc ; I_batt = P_batt / V_nom."""
    if power_w <= 0:
        return 0.0
    return (power_w / DCDC_EFFICIENCY) / V_NOMINAL


def peukert_capacity_ah(profile: DeviceProfile) -> float:
    """C' = C * (C/R)^(n-1), la capacita' effettiva alla corrente nominale."""
    c_rated_ah = profile.battery_capacity_mah / 1000.0
    i_rated_a = c_rated_ah / RATED_R_HOURS
    return c_rated_ah * (i_rated_a ** (profile.peukert_n - 1.0))


def delta_soc(power_w: float, dt_s: float, profile: DeviceProfile) -> float:
    """Variazione di SoC (<= 0) consumando power_w per dt_s secondi."""
    if power_w <= 0 or dt_s <= 0:
        return 0.0
    current_a = load_power_to_battery_current(power_w)
    t_full_h = peukert_capacity_ah(profile) / (current_a ** profile.peukert_n)
    return -((dt_s / 3600.0) / t_full_h)


def energy_wh_from_delta_soc(delta: float, profile: DeviceProfile) -> float:
    """Energia estratta DALLA BATTERIA, non lavoro utile del device.

    Con Peukert il ΔSoC per un dato lavoro dipende dalla corrente, quindi
    questa e' la carica che la batteria paga. E' la definizione giusta per
    una tesi battery-aware, ma va dichiarata: non e' l'energia consumata dal
    calcolo.
    """
    c_rated_ah = profile.battery_capacity_mah / 1000.0
    return V_NOMINAL * (-delta * c_rated_ah)


def c_rate(current_a: float, profile: DeviceProfile) -> float:
    return current_a / (profile.battery_capacity_mah / 1000.0)


if __name__ == "__main__":
    from device.constants import TRAIN_POWER_W
    from device.device_profile import generate_profiles

    p = generate_profiles(1)[0]
    i = load_power_to_battery_current(TRAIN_POWER_W)
    print(f"batteria {p.battery_capacity_mah:.0f} mAh, n={p.peukert_n}")
    print(f"P={TRAIN_POWER_W} W -> I={i:.3f} A ({c_rate(i, p):.2f}C)")
    print(f"C' = {peukert_capacity_ah(p):.4f} Ah "
          f"(nominale {p.battery_capacity_mah/1000:.3f} Ah)")
    t_full_h = peukert_capacity_ah(p) / (i ** p.peukert_n)
    print(f"scarica piena a questa corrente: {t_full_h:.3f} h")
    for dt in (10.0, 32.6, 60.0):
        d = delta_soc(TRAIN_POWER_W, dt, p)
        print(f"  {dt:5.1f} s -> dSoC {d*100:+.3f}%  "
              f"({energy_wh_from_delta_soc(d, p)*1000:.2f} mWh)")

    # Verifica formula, Esempio 5.2 handout: C=1Ah, n=1.1, R=5h -> C'=0.851 Ah
    class _Fake:
        battery_capacity_mah = 1000.0
        peukert_n = 1.1
    cp = peukert_capacity_ah(_Fake())  # type: ignore[arg-type]
    print(f"\n[check handout Es.5.2] atteso 0.851 Ah -> {cp:.3f} Ah")
    print(f"[check handout Es.5.2] durata a 0.2A attesa 5.00 h -> "
          f"{cp / (0.2 ** 1.1):.2f} h")
