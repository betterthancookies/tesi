"""Modello di batteria da datasheet: V(SoC, I) e Q(I) tabulati.

SOSTITUISCE PEUKERT. Peukert riassume la non-idealita' della batteria in un
solo esponente, che nel simulatore era n=1.15 -- un valore da piombo-acido,
mentre per il litio la letteratura riporta 1.01-1.10. La sensitivity fatta
allora lo confermava: a n=1.05 l'errore del modello lineare scendeva da 1.26x
a 1.08x e gli esaurimenti passavano da 10 a 0. Era il parametro piu' esposto
di tutto il simulatore.

Qui non c'e' nessun esponente. Due tabelle prese dal datasheet:

    V(SoC, C-rate)   curve di scarica a 0.2C, 0.5C, 1C
    Q(C-rate)        capacita' utilizzabile, dagli endpoint delle stesse curve

e la corrente si ricava risolvendo P = V(SoC, I) * I per bisezione.

[B] PERCHE' E' MEGLIO, al di la' del fatto che i valori sono citabili. Peukert
applica una correzione che dipende dalla sola corrente, quindi COSTANTE lungo
tutta la run: 1.07x a SoC 1.0 come a SoC 0.05. Qui la tensione cala al calare
della carica, quindi a potenza costante la corrente sale e la scarica accelera.
L'errore del modello lineare e' massimo ESATTAMENTE dove il vincolo di riserva
conta, il che e' un argomento piu' forte di una sottostima uniforme.

[B] LE CURVE SONO NORMALIZZATE IN C-RATE, non in ampere. Il datasheet e' di una
cella da 2.6 Ah, i device simulati ne hanno 3.5-4.5: usare gli ampere assoluti
legherebbe il modello a quella taglia. Con il C-rate le stesse curve valgono
per qualunque capacita', che e' anche il motivo per cui il C-rate esiste.

FONTE: curve di scarica del datasheet (Ta=25 C), digitalizzate. Gli endpoint
Q(0.2C)=2.60 Ah, Q(0.5C)=2.50 Ah, Q(1C)=2.39 Ah sono letti dalle stesse curve
e qui espressi come frazione della capacita' nominale.
"""

from __future__ import annotations

import numpy as np

from device.constants import DCDC_EFFICIENCY, V_NOMINAL
from device.device_profile import DeviceProfile

# ---------------------------------------------------------------- tabelle
# SoC crescente: 0 = scarica, 1 = piena. Il datasheet riporta la capacita'
# SCARICATA in ascissa, quindi SoC = 1 - x/100 e le curve sono ribaltate.
_SOC = np.array([0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40,
                 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00])

# V a 0.2C -- il ginocchio finale sta intorno al 5-10% di carica residua
_V_02C = np.array([2.50, 2.95, 3.30, 3.45, 3.50, 3.53, 3.57, 3.60,
                   3.65, 3.70, 3.76, 3.85, 3.93, 4.02, 4.18, 4.30])
# V a 0.5C
_V_05C = np.array([2.45, 2.85, 3.20, 3.35, 3.42, 3.46, 3.50, 3.54,
                   3.58, 3.63, 3.70, 3.80, 3.88, 3.97, 4.13, 4.27])
# V a 1C
_V_1C = np.array([2.40, 2.75, 3.05, 3.20, 3.28, 3.33, 3.38, 3.42,
                  3.45, 3.50, 3.57, 3.66, 3.74, 3.83, 3.90, 3.97])

_CRATES = np.array([0.2, 0.5, 1.0])
_VTAB = np.vstack([_V_02C, _V_05C, _V_1C])          # (3, 16)

# capacita' utilizzabile come FRAZIONE della nominale, dagli endpoint
_QFRAC = np.array([2.60, 2.50, 2.39]) / 2.60         # 1.000, 0.962, 0.919

_BISECT_ITERS = 60


def _voltage(soc: float, crate: float) -> float:
    """V(SoC, C-rate), bilineare sulle tre curve.

    [B] fuori dal range 0.2-1C si estrapola LINEARMENTE in C-rate invece di
    saturare: a correnti molto basse (idle, ~0.001C) la tensione satura da
    sola sulla curva 0.2C, che e' fisicamente corretto, mentre a correnti alte
    saturare sottostimerebbe la caduta e falserebbe proprio il caso peggiore.
    """
    soc = float(np.clip(soc, 0.0, 1.0))
    v_at = np.array([np.interp(soc, _SOC, _VTAB[i]) for i in range(3)])
    if crate <= _CRATES[0]:
        # sotto 0.2C la caduta ohmica e' gia' trascurabile: non estrapolare
        # verso l'alto, la tensione a vuoto e' il limite fisico
        return float(v_at[0])
    if crate >= _CRATES[-1]:
        slope = (v_at[2] - v_at[1]) / (_CRATES[2] - _CRATES[1])
        return float(max(0.5, v_at[2] + slope * (crate - _CRATES[-1])))
    return float(np.interp(crate, _CRATES, v_at))


def _q_fraction(crate: float) -> float:
    """Capacita' utilizzabile / nominale, in funzione del C-rate."""
    if crate <= _CRATES[0]:
        return float(_QFRAC[0])
    if crate >= _CRATES[-1]:
        slope = (_QFRAC[2] - _QFRAC[1]) / (_CRATES[2] - _CRATES[1])
        return float(max(0.3, _QFRAC[2] + slope * (crate - _CRATES[-1])))
    return float(np.interp(crate, _CRATES, _QFRAC))


def solve_current_a(power_w: float, soc: float, profile: DeviceProfile) -> float:
    """Risolve P = V(SoC, I) * I per bisezione. Ritorna I in ampere.

    [B] serve un metodo iterativo perche' V dipende da I: piu' corrente, meno
    tensione, quindi per la stessa potenza serve ancora piu' corrente. Il
    prodotto V*I e' monotono crescente in I finche' la caduta resta moderata,
    il che rende la bisezione sicura e sufficiente -- Newton converge prima ma
    puo' uscire dal dominio tabulato.
    """
    if power_w <= 0.0:
        return 0.0
    q_nom_ah = profile.battery_capacity_mah / 1000.0
    p_batt = power_w / DCDC_EFFICIENCY        # la conversione avviene a monte
    lo, hi = 1e-9, 50.0 * q_nom_ah            # fino a 50C: limite di sicurezza
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        if _voltage(soc, mid / q_nom_ah) * mid < p_batt:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def delta_soc(power_w: float, dt_s: float, profile: DeviceProfile,
              soc: float = 1.0) -> float:
    """Variazione di SoC (<= 0) consumando power_w per dt_s secondi.

    [B] `soc` ha default 1.0 SOLO per compatibilita' con le chiamate che non
    lo passano ancora. Passarlo cambia il risultato -- e' tutto il punto del
    modello -- quindi ogni chiamante dovrebbe fornire il SoC corrente.
    """
    if power_w <= 0.0 or dt_s <= 0.0:
        return 0.0
    q_nom_ah = profile.battery_capacity_mah / 1000.0
    i_a = solve_current_a(power_w, soc, profile)
    q_eff_ah = q_nom_ah * _q_fraction(i_a / q_nom_ah)
    return -(i_a * (dt_s / 3600.0)) / q_eff_ah


def load_power_to_battery_current(power_w: float,
                                  profile: DeviceProfile | None = None,
                                  soc: float = 1.0) -> float:
    """Corrente assorbita. Senza profilo ricade su V nominale (compatibilita')."""
    if power_w <= 0.0:
        return 0.0
    if profile is None:
        return (power_w / DCDC_EFFICIENCY) / V_NOMINAL
    return solve_current_a(power_w, soc, profile)


def energy_wh_linear(power_w: float, dt_s: float) -> float:
    """Energia SENZA il modello di batteria: P/eta * dt.

    E' la contabilita' dei paper di riferimento, che stimano la carica residua
    sottraendo i Wh spesi dalla capacita' nominale. Invariata: e' il termine di
    paragone, non deve seguire il modello nuovo.
    """
    if power_w <= 0.0 or dt_s <= 0.0:
        return 0.0
    return (power_w / DCDC_EFFICIENCY) * dt_s / 3600.0


def energy_wh_from_delta_soc(delta: float, profile: DeviceProfile) -> float:
    """Energia estratta DALLA BATTERIA per una data variazione di SoC."""
    q_nom_ah = profile.battery_capacity_mah / 1000.0
    return V_NOMINAL * (-delta * q_nom_ah)


def c_rate(current_a: float, profile: DeviceProfile) -> float:
    return current_a / (profile.battery_capacity_mah / 1000.0)


def model_error_vs_linear(power_w: float, dt_s: float, profile: DeviceProfile,
                          soc: float) -> float:
    """Rapporto fra consumo reale e stima lineare, a un dato SoC.

    E' la grandezza che il capitolo sulla contabilita' energetica misura:
    quanto il modello dei paper sottostima. Con Peukert era costante; qui
    cresce al calare della carica.
    """
    lin = energy_wh_linear(power_w, dt_s)
    if lin <= 0.0:
        return 1.0
    real = energy_wh_from_delta_soc(delta_soc(power_w, dt_s, profile, soc), profile)
    return real / lin


if __name__ == "__main__":
    from device.constants import TRAIN_POWER_W
    from device.device_profile import generate_profiles

    p = generate_profiles(1)[0]
    q = p.battery_capacity_mah / 1000.0
    print(f"cella {p.battery_capacity_mah:.0f} mAh, P_train {TRAIN_POWER_W} W\n")
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
    i = solve_current_a(5.0, 0.50, _Cell())          # type: ignore[arg-type]
    d = delta_soc(5.0, 60.0, _Cell(), 0.50)          # type: ignore[arg-type]
    print(f"  I = {i:.3f} A ({i/2.6:.2f}C), V = {_voltage(0.5, i/2.6):.3f} V, "
          f"V*I = {_voltage(0.5, i/2.6)*i:.2f} W")
    print(f"  dSoC su 60 s = {d:+.5f}  (documento: -0.00816)")