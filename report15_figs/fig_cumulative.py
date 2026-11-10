"""Ablazione cumulata SAGE-smart / SAGE-smart 2 a beta 0.1, stile della figura 2 del report 13.

Stessi quattro pannelli (accuratezza finale = media degli ultimi 20 round,
energia, telefoni scarichi a fine run, SoC medio finale), stesse linee di
riferimento (FedAvg e SAGE ablation della campagna v2, beta 0.1, seed 2026).
Qui un beta solo, quindi le barre sono i valori del beta 0.1 (niente punti).

    python report15_figs/fig_cumulative.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUM = os.path.join(ROOT, "tesifl", "results_ablation_cum_smart2_ResNet18_C100")
MAIN = os.path.join(ROOT, "tesifl", "results_rechargeONdeviceON_ResNet18_C100_v2")
OUT = os.path.join(ROOT, "report15_figs")
B = "01"

PURPLE, DARK_PURPLE = "#9467bd", "#5b3a8c"
TEAL, DARK_TEAL = "#3fa7a3", "#1f6e6b"
# (etichetta sull'asse, file, colore, posizione x)
steps = [
    ("0 base", "sage_smart_cum0_nm_x1", "#bbbbbb", 0.0),
    ("1 +staleness", "sage_smart_cum1_nm_x1", PURPLE, 1.0),
    ("2 +adaptive E", "sage_smart_cum2_nm_x1", PURPLE, 2.0),
    ("3 +small B", "sage_smart_cum3_nm_x1", PURPLE, 3.0),
    ("4 +partial work", "sage_smart_pcum4_nm_x1", PURPLE, 4.0),
    ("5 +negotiation", "sage_smart_pcum5_nm_x1", PURPLE, 5.0),
    ("6 +SoC estimate\n(SAGE-smart)", "sage_smart_plus4_nm_x1", DARK_PURPLE, 6.0),
    ("5' -broadcast", "sage_smart2_noinfo_nm_x1", TEAL, 7.4),
    ("6' +negotiation 2", "sage_smart2_nofc_nm_x1", TEAL, 8.4),
    ("7' +SoC forecast\n(SAGE-smart 2)", "sage_smart2_socfc_nm_x1", DARK_TEAL, 9.4),
]
ref = {"FedAvg (linear battery)": os.path.join(MAIN, f"fedavg_b{B}_seed2026.csv"),
       "SAGE ablation": os.path.join(MAIN, f"sage_nm_b{B}_seed2026.csv")}


def final(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    acc = [float(r["accuracy"]) for r in rows]
    r = rows[-1]
    return (100 * sum(acc[-20:]) / 20, float(r["total_energy_wh"]),
            int(float(r["n_soc_zero"])), 100 * float(r["mean_soc"]))


metrics = ["Final accuracy (%)", "Energy (Wh)", "Depleted devices", "Mean final SoC (%)"]
fmt = ["{:.1f}", "{:.0f}", "{:d}", "{:.1f}"]
vals = {}
for name, lab, _, _ in steps:
    p = os.path.join(CUM, f"{lab}_b{B}_seed2026.csv")
    vals[name] = final(p) if os.path.exists(p) else None
refv = {name: final(p) for name, p in ref.items()}

fig, ax = plt.subplots(1, 4, figsize=(19, 4.8))
for k in range(4):
    a = ax[k]
    for name, _, col, x in steps:
        v = vals[name]
        if v is None:      # run non ancora finita: segnaposto (coordinate dell'asse)
            a.text(x, 0.03, "running", rotation=90, ha="center", va="bottom", fontsize=8,
                   color="#777777", transform=a.get_xaxis_transform())
            continue
        a.bar(x, v[k], color=col, alpha=0.85)
        a.text(x, v[k], fmt[k].format(v[k]), ha="center", va="bottom", fontsize=7)
    for (rname, rv), rc in zip(refv.items(), ("#444444", "#e377c2")):
        a.axhline(rv[k], color=rc, ls="--", lw=1.2, label=rname)
    a.axvline(6.7, color="#999999", ls=":", lw=1)       # fra i due rami
    a.set_xticks([x for _, _, _, x in steps])
    a.set_xticklabels([n for n, _, _, _ in steps], rotation=45, ha="right", fontsize=8)
    a.set_title(metrics[k])
    a.grid(axis="y", alpha=0.3)
ax[0].set_ylim(30, 60)
ax[3].set_ylim(0, 50)
ax[2].legend(fontsize=7.5, loc="center right")
fig.suptitle("Cumulative ablation of SAGE-smart (purple) and SAGE-smart 2 (teal) from the same step 4 "
             "(datasheet battery, $\\beta$ = 0.1, seed 2026): dashed = references")
fig.tight_layout()
os.makedirs(OUT, exist_ok=True)
out = os.path.join(OUT, "fig_cumulative_ablation_b01.png")
fig.savefig(out, dpi=150)

print(f"{'gradino':34s}   acc     Wh  scarichi  SoC finale")
for name, _, _, _ in steps:
    v = vals[name]
    label = name.replace("\n", " ")
    print(f"  {label:32s} " + ("  in corso" if v is None else
                               f"{v[0]:5.1f} {v[1]:6.0f} {v[2]:6d} {v[3]:8.1f}"))
for name, v in refv.items():
    print(f"  {name:32s} {v[0]:5.1f} {v[1]:6.0f} {v[2]:6d} {v[3]:8.1f}")
print("figura:", out)
