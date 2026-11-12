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
# [10/11] solo SAGE-smart 2, con la scala del report 13 chiesta dall'utente
# (0 base, 1 +c, 2 +E adattivo, 3 +B adattivo, 4 +negoziazione, 5 +stima SoC)
# piu' il lavoro parziale come gradino a se': fa parte di SAGE-smart 2 e
# senza il suo gradino il suo guadagno (+2.7 punti) finirebbe sulla
# negoziazione. La negoziazione di SAGE-smart 2 (rosa di 1.5 k, scelta
# fresca) prende il posto del broadcast dello stato: 4 -> 5 e' pcum4 -> nofc.
# [10/11] l'utente vuole ESATTAMENTE i sei gradini del report 13. Il lavoro
# parziale entra nell'E adattivo (il client decide quanto lavorare, anche
# meno di un'epoca), cosi' ogni gradino e' un solo componente:
# 2 = cum2p (cum2 + lavoro parziale), 3 = pcum4 (cum3 + lavoro parziale),
# 4 = nofc (negoziazione, che prende il posto del broadcast), 5 = socfc.
steps2 = [
    ("0 base", "sage_smart_cum0_nm_x1", "#bbbbbb", 0.0),
    ("1 +staleness (c)", "sage_smart_cum1_nm_x1", TEAL, 1.0),
    ("2 +adaptive E", "sage_smart_cum2p_nm_x1", TEAL, 2.0),
    ("3 +adaptive B", "sage_smart_pcum4_nm_x1", TEAL, 3.0),
    ("4 +negotiation", "sage_smart2_nofc_nm_x1", TEAL, 4.0),
    ("5 +SoC estimate\n(SAGE-smart 2)", "sage_smart2_socfc_nm_x1", DARK_TEAL, 5.0),
]
NOTE2 = ("Adaptive E includes partial work: a client that cannot meet the deadline even with E = 2 "
         "sends the SGD steps that fit, weighted by the samples seen. Steps 0-3 poll the state "
         "with a broadcast; from step 4 the negotiation replaces it.")
# [11/11] l'utente unisce gli ultimi due gradini (negoziazione e stima lavorano
# insieme: la stima decide a chi chiedere, la negoziazione sceglie fra chi
# accetta): il 4 e' direttamente SAGE-smart 2 (socfc). Ai tre beta (seed 2026)
# come la figura del report 13: barre = media sui beta presenti per TUTTI i
# gradini, punti = ogni beta.
steps3 = [
    ("0 base", "sage_smart_cum0_nm_x1", "#bbbbbb", 0.0),
    ("1 +staleness (c)", "sage_smart_cum1_nm_x1", TEAL, 1.0),
    ("2 +adaptive E", "sage_smart_cum2p_nm_x1", TEAL, 2.0),
    ("3 +adaptive B", "sage_smart_pcum4_nm_x1", TEAL, 3.0),
    ("4 +negotiation\n+SoC estimate\n(SAGE-smart 2)", "sage_smart2_socfc_nm_x1", DARK_TEAL, 4.0),
]
NOTE3 = ("Adaptive E includes partial work: a client that cannot meet the deadline even with E = 2 "
         "sends the SGD steps that fit, weighted by the samples seen.\nStep 4 adds negotiation and "
         "server-side SoC estimate together: the estimate ranks whom to ask, the negotiation (which "
         "replaces the state broadcast of steps 0-3) picks among those who accept.")
BETAS = (("01", "0.1", "#1f77b4"), ("05", "0.5", "#ff7f0e"), ("10", "1.0", "#2ca02c"))
REF = {"FedAvg (linear battery)": os.path.join(MAIN, "fedavg_b{b}_seed2026.csv"),
       "SAGE ablation": os.path.join(MAIN, "sage_nm_b{b}_seed2026.csv")}
ref = {name: p.format(b=B) for name, p in REF.items()}


def final(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    acc = [float(r["accuracy"]) for r in rows]
    r = rows[-1]
    return (100 * sum(acc[-20:]) / 20, float(r["total_energy_wh"]),
            int(float(r["n_soc_zero"])), 100 * float(r["mean_soc"]))


metrics = ["Final accuracy (%)", "Energy (Wh)", "Depleted devices", "Mean final SoC (%)"]
fmt = ["{:.1f}", "{:.0f}", "{:d}", "{:.1f}"]
refv = {name: final(p) for name, p in ref.items()}


def draw(steps, fname, title, sep=None, figsize=(19, 4.8), note=None):
    vals = {}
    for name, lab, _, _ in steps:
        p = os.path.join(CUM, f"{lab}_b{B}_seed2026.csv")
        vals[name] = final(p) if os.path.exists(p) else None
    fig, ax = plt.subplots(1, 4, figsize=figsize)
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
        if sep is not None:
            a.axvline(sep, color="#999999", ls=":", lw=1)       # fra i due rami
        a.set_xticks([x for _, _, _, x in steps])
        a.set_xticklabels([n for n, _, _, _ in steps], rotation=45, ha="right", fontsize=8)
        a.set_title(metrics[k])
        a.grid(axis="y", alpha=0.3)
    ax[0].set_ylim(30, 60)
    ax[3].set_ylim(0, 50)
    ax[2].legend(fontsize=7.5, loc="center right")
    fig.suptitle(title)
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8.5, color="#444444")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    else:
        fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"{'gradino':34s}   acc     Wh  scarichi  SoC finale")
    for name, _, _, _ in steps:
        v = vals[name]
        label = name.replace("\n", " ")
        print(f"  {label:32s} " + ("  in corso" if v is None else
                                   f"{v[0]:5.1f} {v[1]:6.0f} {v[2]:6d} {v[3]:8.1f}"))
    for name, v in refv.items():
        print(f"  {name:32s} {v[0]:5.1f} {v[1]:6.0f} {v[2]:6d} {v[3]:8.1f}")
    print("figura:", out, "\n")


def draw_multi(steps, fname, note=None, figsize=(13, 5.2), it=False):
    """Come la figura del report 13: barre = media sui beta, punti = ogni beta.

    Le barre usano solo i beta presenti per TUTTI i gradini (una barra non
    deve mediare su beta diversi dalle altre); con un beta solo niente punti.
    """
    vals = {}
    for name, lab, _, _ in steps:
        vals[name] = {}
        for b, _, _ in BETAS:
            p = os.path.join(CUM, f"{lab}_b{b}_seed2026.csv")
            if os.path.exists(p):
                vals[name][b] = final(p)
    common = [b for b, _, _ in BETAS if all(b in v for v in vals.values())]
    if not common:
        print("nessun beta comune a tutti i gradini:", {n: sorted(v) for n, v in vals.items()})
        return
    # [12/11] solo FedAvg come riferimento (la linea di SAGE ablation tolta)
    refs = {name: [final(p.format(b=b)) for b in common] for name, p in REF.items()
            if name.startswith("FedAvg")}
    shown = ", ".join(bl for b, bl, _ in BETAS if b in common)
    fig, ax = plt.subplots(1, 4, figsize=figsize)
    xs = [x for _, _, _, x in steps]
    for k in range(4):
        a = ax[k]
        for name, _, col, x in steps:
            m = sum(vals[name][b][k] for b in common) / len(common)
            a.bar(x, m, color=col, alpha=0.85)
            # etichetta col fondo bianco sopra i punti, cosi' resta leggibile
            txt = fmt[k].format(round(m) if k == 2 else m)
            a.text(x, m, txt.replace(".", ",") if it else txt, ha="center", va="bottom",
                   fontsize=7, zorder=5, bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                                   ec="none", alpha=0.8))
        if len(common) > 1:
            for j, (b, bl, bc) in enumerate(BETAS):
                if b in common:      # punti sfalsati per beta, come colonne affiancate
                    a.scatter([x + 0.25 * (j - 1) for x in xs],
                              [vals[n][b][k] for n, _, _, _ in steps], s=14, zorder=3,
                              color=bc, label=f"$\\beta$={bl.replace('.', ',') if it else bl}")
        for (rname, rv), rc in zip(refs.items(), ("#444444", "#e377c2")):
            a.axhline(sum(v[k] for v in rv) / len(rv), color=rc, ls="--", lw=1.2,
                      label="FedAvg (batteria lineare)" if it else rname)
        a.set_xticks(xs)
        a.set_xticklabels([n for n, _, _, _ in steps], rotation=45, ha="right", fontsize=8)
        a.set_title(METRICS_IT[k] if it else metrics[k])
        a.grid(axis="y", alpha=0.3)
    ax[0].set_ylim(30, 62)
    ax[3].set_ylim(0, 50)
    ax[2].legend(fontsize=7.5, loc="center right")
    what = ("bars = mean over the three $\\beta$, dots = each $\\beta$" if len(common) == 3
            else (f"bars = mean over $\\beta$ = {shown}, dots = each $\\beta$" if len(common) > 1
                  else f"$\\beta$ = {shown}"))
    if it:
        what = ("barre = media sui tre $\\beta$, punti = ogni $\\beta$" if len(common) == 3
                else f"$\\beta$ = {shown}".replace(".", ","))
        fig.suptitle(f"Ablazione cumulata di SAGE-smart 2 (batteria da datasheet, seed 2026): {what}")
    else:
        fig.suptitle(f"Cumulative ablation of SAGE-smart 2 (datasheet battery, seed 2026): {what}")
    if note:
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8.5, color="#444444")
        fig.tight_layout(rect=(0, 0.08, 1, 1))
    else:
        fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"beta nelle barre: {shown}")
    print(f"{'gradino':34s}   acc     Wh  scarichi  SoC finale   (per beta: acc)")
    for name, _, _, _ in steps:
        v = vals[name]
        mm = [sum(v[b][k] for b in common) / len(common) for k in range(4)]
        per = "  ".join(f"b{b}:{v[b][0]:.1f}" for b, _, _ in BETAS if b in v)
        print(f"  {name.replace(chr(10), ' '):32s} {mm[0]:5.1f} {mm[1]:6.0f} {mm[2]:6.1f} {mm[3]:8.1f}   ({per})")
    print("figura:", out, "\n")


draw(steps, "fig_cumulative_ablation_b01.png",
     "Cumulative ablation of SAGE-smart (purple) and SAGE-smart 2 (teal) from the same step 4 "
     "(datasheet battery, $\\beta$ = 0.1, seed 2026): dashed = references", sep=6.7)
draw_multi(steps3, "fig_cumulative_ablation_smart2.png", figsize=(13, 4.6))
# [12/11] la stessa scala in ORDINE INVERSO (negoziazione e stima subito dopo
# l'anzianita'), beta 0.1: stesso inizio e stessa fine (socfc)
steps_rev = [
    ("0 base", "sage_smart_cum0_nm_x1", "#bbbbbb", 0.0),
    ("1 +staleness (c)", "sage_smart_cum1_nm_x1", TEAL, 1.0),
    ("2 +negotiation\n+SoC estimate", "sage_smart2_rev2_nm_x1", TEAL, 2.0),
    ("3 +adaptive E", "sage_smart2_rev3_nm_x1", TEAL, 3.0),
    ("4 +adaptive B\n(SAGE-smart 2)", "sage_smart2_socfc_nm_x1", DARK_TEAL, 4.0),
]
draw_multi(steps_rev, "fig_cumulative_ablation_smart2_reverse.png", figsize=(13, 4.6))
METRICS_IT = ["Accuratezza finale (%)", "Energia (Wh)", "Dispositivi scarichi", "SoC medio finale (%)"]
