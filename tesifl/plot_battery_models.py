#!/usr/bin/env python3
"""Pareto SoC / accuracy per modello di batteria, uno per algoritmo.

Da ~/tesifl (dopo una campagna con le etichette di experiment.toml):

    python plot_battery_models.py --dir results_<name>
    python plot_battery_models.py --dir results_<name> --beta 0.5
    python plot_battery_models.py --dir results_<name> --path

Per SAGE, SAGE-smart ed ESCS-SD mette sullo stesso piano i tre bracci che
differiscono SOLO per il modello di batteria con cui l'algoritmo stima il SoC
(lin e peuk sono fuel gauge completi: idle e ricarica inclusi):

                lineare           Peukert            datasheet (nm)
    SAGE        sage_lin          sage_peuk          sage_soc
    SAGE-smart  sage_smart_lin    sage_smart_peuk    sage_smart
    ESCS-SD     escs_sd_lin       escs_sd_peuk       escs_sd

e, se presente, in grigio la contabilita' dei paper (solo i round in cui il
client lavora, idle ignorato): sage per SAGE, escs_sd_paper per ESCS-SD.

Il grafico e' quello di fig5 di plot_curves.py (fig_pareto_soc):
    x  SoC medio a fine run, morti contati come 0
    y  accuracy media degli ultimi TAIL round
    marker piu' grande = piu' client esauriti (†)
stessi assi, stessi marker, stesso posizionamento delle etichette. Cambia
solo il colore, che qui indica il MODELLO (uguale nei tre algoritmi):
grigio paper, rosso lineare, arancio Peukert, verde datasheet.

Produce in --dir, con suffisso _ball (tutti i beta) o _b<beta>:
    figB_pareto_soc_sage_<b>.png
    figB_pareto_soc_sage_smart_<b>.png
    figB_pareto_soc_escs_sd_<b>.png
    figB_pareto_soc_models_<b>.png      i tre affiancati, stessa scala y

[B] DEFAULT: tutti i beta insieme (media su tutte le run, beta x seed), come
la serie _ball di plot_curves, ma per ogni algoritmo SOLO sui beta presenti in
tutte e tre le varianti: se una manca di un beta, mediare il resto
confronterebbe regimi diversi. Le run escluse vengono stampate. Come li' i tre beta sono regimi diversi: la
media serve al quadro d'insieme, le conclusioni vanno verificate per beta
(--beta 0.1 / 0.5 / 1.0). Nella serie pooled le barre d'errore non si
disegnano, perche' sarebbero la dispersione FRA beta e non incertezza.

--path collega i tre punti di ogni algoritmo nell'ordine lin -> peuk -> nm:
la direzione dice cosa cambia migliorando il modello di batteria.
"""
import argparse
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_curves as pc

FAMILIES = [
    ("sage", "SAGE",
     {"paper": "sage", "lin": "sage_lin", "peuk": "sage_peuk", "nm": "sage_soc"}),
    ("sage_smart", "SAGE-smart",
     {"lin": "sage_smart_lin", "peuk": "sage_smart_peuk", "nm": "sage_smart"}),
    ("escs_sd", "ESCS-SD",
     {"paper": "escs_sd_paper", "lin": "escs_sd_lin", "peuk": "escs_sd_peuk",
      "nm": "escs_sd"}),
]
# "paper" = contabilita' dei paper (idle ignorato), in grigio: e' un
# riferimento, il confronto sul modello di batteria e' lin -> peuk -> nm
MODELS = [("paper", "paper", "#7f7f7f"),
          ("lin", "lin", "#d62728"),
          ("peuk", "peuk", "#ff7f0e"),
          ("nm", "nm", "#2ca02c")]


def collect(results_dir, beta):
    """label -> {btag -> lista di (SoC finale, accuracy di coda, morti)}.

    Stesso calcolo di plot_curves.fig_pareto_soc, ma sulle etichette singole
    (il raggruppamento di plot_curves fonderebbe escs_sd con escs_sp/md/mp)
    e tenendo il beta di ogni run, per poter mediare sui soli beta comuni.
    """
    want = None if beta == pc.ALL else pc.btag(beta)
    stats = {}
    for path in glob.glob(os.path.join(results_dir, "*_seed*.csv")):
        m = pc.PAT_BETA.match(os.path.basename(path))
        if not m:
            continue
        lab, tag = m.group(1), m.group(2)
        if want is not None and tag != want:
            continue
        with open(path) as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("mean_soc")]
        ar = pc._acc_rows(rows)
        if not rows or not ar:
            continue
        stats.setdefault(lab, {}).setdefault(tag, []).append(
            (float(rows[-1]["mean_soc"]),
             float(np.mean([x[1] for x in ar[-pc.TAIL:]])),
             float(rows[-1]["n_soc_zero"])))
    return stats


def common_betas(stats, labels):
    """Beta presenti in TUTTE le varianti disponibili dell'algoritmo.

    [B] mediare su beta diversi confronterebbe regimi diversi: se ESCS-nm ha
    solo beta 1.0 e ESCS-lin tutti e tre, la differenza fra i due punti
    sarebbe in gran parte l'effetto del beta, non del modello di batteria.
    """
    # solo lin / peuk / nm: il braccio paper e' un riferimento e non deve
    # restringere il confronto principale (vedi restrict)
    sets = [set(stats[lab]) for key, lab in labels.items()
            if key != "paper" and lab in stats]
    return sorted(set.intersection(*sets)) if sets else []


def restrict(stats, labels, betas):
    """label -> lista di punti, sui soli beta indicati.

    Il braccio paper entra solo se ha TUTTI quei beta, altrimenti la sua media
    sarebbe su regimi diversi da quella degli altri tre punti.
    """
    out = {}
    for key, lab in labels.items():
        if lab not in stats:
            continue
        if key == "paper" and not all(b in stats[lab] for b in betas):
            continue
        out[lab] = [p for b in betas for p in stats[lab].get(b, [])]
    return out


def draw(ax, stats, name, labels, beta, path=False):
    """Un Pareto come fig5 per un algoritmo. Ritorna i modelli disegnati."""
    bars = beta != pc.ALL
    items, pts, drawn = [], [], []
    for key, short, color in MODELS:
        runs = stats.get(labels.get(key))
        if not runs:
            continue
        S = [v[0] for v in runs]
        A = [v[1] for v in runs]
        D = float(np.mean([v[2] for v in runs]))
        mx, my = float(np.mean(S)), float(np.mean(A))
        ax.errorbar(mx, my, xerr=np.std(S) if bars else None,
                    yerr=np.std(A) if bars else None,
                    fmt="o", ms=6 + 2.5 * np.sqrt(D), capsize=3 if bars else 0,
                    alpha=0.85, color=color, label=f"{name} ({short})")
        tag = f"{name} ({short})"
        if D > 0:
            tag += f" ({D:.0f}†)"
        items.append((mx, my, tag, color))
        if key != "paper":
            pts.append((mx, my))     # --path: solo lin -> peuk -> nm
        drawn.append(key)
    if path and len(pts) > 1:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            ax.annotate("", (x1, y1), (x0, y0),
                        arrowprops=dict(arrowstyle="->", color="#888888",
                                        lw=1.0, ls="--", shrinkA=8, shrinkB=8))
    ax.set_xlabel("Mean state of charge at end of run (dead clients counted as 0)")
    ax.set_ylabel(f"Test accuracy (mean of last {pc.TAIL} rounds)")
    ax.grid(alpha=0.3)
    ax.margins(0.16)
    pc._place_labels(ax, items)
    return drawn


def beta_label(beta, betas):
    """Beta mostrati nel titolo: 'all β pooled' solo se ci sono davvero tutti."""
    if beta != pc.ALL:
        return pc.beta_str(beta)
    shown = ", ".join(b[0] + "." + b[1:] if len(b) == 2 else b for b in betas)
    return "all β pooled" if len(betas) >= 3 else f"{shown} only (common to all variants)"


def title(name, beta, betas):
    return (f"Survival / accuracy at end of run — {name} — "
            f"β = {beta_label(beta, betas)}\n"
            "larger marker = more depleted clients (†); top-right is better")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--beta", default=pc.ALL,
                    help="'all' (default, tutti i beta insieme) oppure 0.1 / 0.5 / 1.0")
    ap.add_argument("--path", action="store_true",
                    help="collega i punti lin -> peuk -> nm di ogni algoritmo")
    args = ap.parse_args()

    raw = collect(args.dir, args.beta)
    if not raw:
        raise SystemExit(f"nessun CSV in {args.dir} per beta={args.beta}")
    tag = pc.btag(args.beta)
    present = []
    for key, name, labels in FAMILIES:
        betas = common_betas(raw, labels)
        if not betas:
            continue
        present.append((key, name, labels, betas, restrict(raw, labels, betas)))
        # quali run restano fuori perche' il loro beta manca a un'altra variante
        left_out = [f"{lab} b{b}" for lab in labels.values() if lab in raw
                    for b in sorted(raw[lab]) if b not in betas]
        if left_out:
            print(f"[{name}] media sui soli beta comuni {betas}; escluse: "
                  + ", ".join(left_out))
    if not present:
        raise SystemExit("nessuna delle etichette lin/peuk/nm e' presente: "
                         "vedi la tabella in experiment.toml")

    for key, name, labels, betas, stats in present:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        drawn = draw(ax, stats, name, labels, args.beta, args.path)
        ax.set_title(title(name, args.beta, betas))
        fig.tight_layout()
        out = os.path.join(args.dir, f"figB_pareto_soc_{key}_b{tag}.png")
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        missing = [m for m in ("lin", "peuk", "nm") if m not in drawn]
        print(f"  {out}" + (f"   (mancano: {', '.join(labels[m] for m in missing)})"
                            if missing else ""))

    # i tre affiancati, stessa scala y per confrontare gli algoritmi
    fig, axes = plt.subplots(1, len(present), figsize=(7.2 * len(present), 5.8),
                             sharey=True, squeeze=False)
    for ax, (key, name, labels, betas, stats) in zip(axes[0], present):
        draw(ax, stats, name, labels, args.beta, args.path)
        ax.set_title(name if len(betas) >= 3 or args.beta != pc.ALL
                     else f"{name} (β = {beta_label(args.beta, betas)})")
    for ax in axes[0][1:]:
        ax.set_ylabel("")
    fig.suptitle(f"Battery model seen by the algorithm: paper accounting (grey) / "
                 f"linear / Peukert / datasheet (nm) — β = {pc.beta_str(args.beta)}\n"
                 "larger marker = more depleted clients (†); "
                 "top-right is better", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(args.dir, f"figB_pareto_soc_models_b{tag}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


if __name__ == "__main__":
    main()
