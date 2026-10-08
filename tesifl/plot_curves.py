#!/usr/bin/env python3
"""Grafici dal CSV della curva energia-accuratezza.

Da ~/tesifl (dopo analyze_rounds.py):  python plot_curves.py

Titoli, assi ed etichette delle figure sono in INGLESE: finiscono in tesi e
nelle slide. I commenti restano in italiano.

Produce in results/, UNA SERIE PER OGNI BETA (suffisso _b01, _b05, _b10):
  fig1_energy_accuracy_b*.png  accuracy in funzione dell'energia spesa.
  fig2_accuracy_rounds_b*.png  accuracy in funzione del round.
  fig3_pareto_b*.png           energia totale vs accuracy di coda.
  fig4_soc_b*.png              SoC a fine run, con morti e mediana.
  fig5_pareto_soc_b*.png       accuracy di coda vs SoC medio A FINE RUN.
  fig6_peak_soc_b*.png         accuracy MASSIMA vs SoC medio NEL ROUND DEL
                               PICCO, col round annotato.
  fig7_thr_soc_b*.png          SoC medio nel round in cui si raggiunge una
                               SOGLIA comune di accuracy.
  fig8_accounting_b*.png       frecce batteria lineare -> datasheet nel
                               piano (morti, accuracy).
  fig9_reserve_b*.png          SoC del client piu' scarico e morti CUMULATI,
                               round per round: mostra se la riserva tiene.

RAGGRUPPAMENTO DELLE VARIANTI
Il suffisso di ogni etichetta e' la batteria dei device (lin, peuk, nm:
lineare, Peukert, datasheet). Le quattro varianti di ESCS sono collassate in
una curva per batteria, la media di {sd, sp, md, mp}:
    ESCS (lin)    i quattro escs_*_lin
    ESCS (peuk)   i quattro escs_*_peuk
    ESCS (nm)     i quattro escs_*_nm
SAGE e SAGE-smart restano separati: sage_lin (il SAGE del paper),
sage_peuk e sage_nm (l'ablazione), sage_smart_lin / _peuk / _nm.

[B] con device omogenei escs_sd ed escs_md danno risultati IDENTICI, e cosi'
escs_sp ed escs_mp: la distinzione system-based / model-based del paper e'
inerte in una popolazione senza eterogeneita' di tempo di training. La media
sulle quattro varianti e' quindi di fatto la media fra deterministico e
probabilistico, con peso doppio ciascuno.

[B] le barre d'errore sui gruppi NON sono varianza fra seed: sono dispersione
fra VARIANTI (e fra seed, se piu' d'uno). Con un seed solo misurano quanto le
varianti differiscono fra loro, il che e' comunque informativo ma va detto.

[B] le varianti ESCS terminano a round diversi (escs_sp si esaurisce prima di
escs_sd). Nelle curve per round la media usa, a ogni round, le sole varianti
ancora vive: dove una si esaurisce la curva del gruppo puo' avere un gradino.
Il pannello dei morti di fig9 lo rende visibile.

--no-group disattiva il raggruppamento e ripristina le dodici curve separate.

SERIE AGGREGATA SUI BETA (suffisso _ball)
Oltre a una serie per ogni beta viene prodotta una serie "pooled" che mette
insieme tutte le beta, con le stesse nove figure.

[B] i tre beta sono REGIMI DIVERSI, non ripetizioni dello stesso esperimento:
a beta 0.1 l'accuratezza di coda sta intorno a 0.45 e a beta 1.0 intorno a
0.55, e il numero di client esauriti cambia di un fattore due. Mediarli
produce un numero che non corrisponde a nessuna condizione sperimentale
reale, e la dispersione che si vede NON e' incertezza ma effetto del beta.
La serie _ball serve a dare un quadro d'insieme in una slide; le conclusioni
vanno tratte dalle serie per beta. Le figure pooled lo dicono in didascalia.

COSA E' L'ENERGIA: la carica estratta dalla batteria, convertita in Wh come
V_nom * C_Ah * (-dSoC), sommata su tutti i client e tutti i round (training,
comunicazione, idle). Il dSoC lo calcola la batteria del mondo della run:
con quella lineare coincide con P*dt, con Peukert le correnti basse costano
meno, col datasheet la stessa energia costa piu' carica a batteria scarica.
E' l'energia che la batteria PAGA, non il lavoro utile prodotto dal device
(quello e' energy_lin_wh nel CSV).

[B] con device identici il SoC medio e' una trasformazione affine ESATTA
dell'energia cumulata:
    SoC_medio = SoC_medio_iniz - E / (N * V_nom * C_Ah)
quindi fig3 e fig5 hanno lo stesso asse x, specchiato e riscalato. Cio' che
il SoC aggiunge e' la DISPERSIONE: a parita' di consumo totale, quanti client
sono morti.

[B] fig5, fig6 e fig7 differiscono per il MOMENTO in cui si misura, e la
scelta non e' neutrale:
  fig5  fine run      -- stesso round per tutti, ma accuracy diverse
  fig6  picco         -- stessa "qualita' migliore", ma round diversi: chi
                         raggiunge il picco presto appare con SoC alto per
                         il solo fatto di aver misurato prima. Il round e'
                         annotato apposta, per rendere visibile la disparita'.
  fig7  soglia comune -- stesso risultato di apprendimento per tutti, quindi
                         il confronto sul SoC e' equo. Chi non raggiunge la
                         soglia esce dal grafico, il che e' informativo.
Fra le tre, fig7 e' quella difendibile in tesi; fig6 e' utile ma va letta
con il round accanto.

Le curve sono la MEDIANA sui seed con banda interquartile.
I beta non si mescolano mai in una stessa figura: sono regimi diversi.
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

# label del CSV -> gruppo mostrato nei grafici. Il suffisso e' la batteria
# dei device, cioe' quella fisica del mondo in cui la run ha girato:
#   lin    lineare     (fedavg, fedprox, sage_lin = SAGE del paper, *_lin)
#   peuk   Peukert     (*_peuk)
#   nm     datasheet   (*_nm)
# Le quattro varianti ESCS confluiscono in un gruppo per batteria.
_ESCS = ("sd", "sp", "md", "mp")
_BATT = ("lin", "peuk", "nm")
GROUP = {
    **{f"escs_{v}_{b}": f"escs_{b}" for v in _ESCS for b in _BATT},
    **{f"sage_{b}": f"sage_{b}" for b in _BATT},
    **{f"sage_smart_{b}": f"sage_smart_{b}" for b in _BATT},
    "fedavg": "fedavg", "fedprox": "fedprox",
}
GROUPED = True          # spento da --no-group


def group_of(label):
    """Nome del gruppo di un'etichetta; identita' se il raggruppamento e' off."""
    if not GROUPED:
        return label
    return GROUP.get(label, label)


ORDER_GROUPED = ["fedavg", "fedprox",
                 "sage_lin", "sage_peuk", "sage_nm",
                 "sage_smart_lin", "sage_smart_peuk", "sage_smart_nm",
                 "escs_lin", "escs_peuk", "escs_nm"]
ORDER = (["fedavg", "fedprox", "sage_lin", "sage_peuk", "sage_nm",
          "sage_smart_lin", "sage_smart_peuk", "sage_smart_nm"]
         + [f"escs_{v}_{b}" for b in _BATT for v in _ESCS])
_BNAME = {"lin": "lin", "peuk": "peuk", "nm": "nm"}
LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx",
         **{f"sage_{b}": f"SAGE ({_BNAME[b]})" for b in _BATT},
         **{f"sage_smart_{b}": f"SAGE-smart ({_BNAME[b]})" for b in _BATT},
         **{f"escs_{b}": f"ESCS ({_BNAME[b]})" for b in _BATT},
         **{f"escs_{v}_{b}": f"ESCS-{v.upper()} ({_BNAME[b]})"
            for v in _ESCS for b in _BATT}}
_ESCS_COLOR = {"sd": "#2ca02c", "sp": "#98df8a", "md": "#17becf", "mp": "#9edae5"}
COLOR = {"fedavg": "#444444", "fedprox": "#1f77b4",
         "sage_lin": "#d62728", "sage_peuk": "#ff9896", "sage_nm": "#e377c2",
         **{f"sage_smart_{b}": "#9467bd" for b in _BATT},
         "escs_lin": "#17becf", "escs_peuk": "#bcbd22", "escs_nm": "#1a7f37",
         **{f"escs_{v}_{b}": c for v, c in _ESCS_COLOR.items() for b in _BATT}}
# tratteggio = lineare, punteggiato = Peukert, continuo = datasheet
_DASH = {"lin": "--", "peuk": ":", "nm": "-"}
STYLE = {**{f"sage_{b}": _DASH[b] for b in _BATT},
         **{f"sage_smart_{b}": _DASH[b] for b in _BATT},
         **{f"escs_{b}": _DASH[b] for b in _BATT},
         **{f"escs_{v}_{b}": _DASH[b] for v in _ESCS for b in _BATT}}
# coppie (batteria lineare, batteria da datasheet): stesso algoritmo, unica
# differenza la batteria dei device. [B] per SAGE la coppia sage_lin ->
# sage_nm cambia anche l'algoritmo (paper -> ablazione).
PAIRS = ([("sage_lin", "sage_nm"), ("sage_smart_lin", "sage_smart_nm")]
         + [(f"escs_{v}_lin", f"escs_{v}_nm") for v in _ESCS])
PAIRS_GROUPED = [("sage_lin", "sage_nm"), ("sage_smart_lin", "sage_smart_nm"),
                 ("escs_lin", "escs_nm")]

TAIL = 20        # round di coda su cui mediare l'accuracy finale
THRESHOLD = 0.40  # soglia di accuracy per fig7

PAT_BETA = re.compile(r"^(.+)_b([0-9]+)_seed([0-9]+)\.csv$")
PAT_PLAIN = re.compile(r"^(.+)_seed([0-9]+)\.csv$")


ALL = "all"          # pseudo-beta: tutte le beta messe insieme


def btag(beta):
    if beta == ALL:
        return "all"
    return "na" if beta == "n/d" else beta.replace(".", "")


def beta_str(beta):
    """Nome del beta da mostrare nei titoli."""
    return "all \u03b2 pooled" if beta == ALL else beta


POOLED_NOTE = ("Pooled across \u03b2 = 0.1, 0.5, 1.0, which are different "
               "regimes rather than repetitions: the spread here is the "
               "effect of \u03b2, not uncertainty. Read the per-\u03b2 "
               "figures for conclusions.")


def _place_labels(ax, items, pad=2.0):
    """Annota i punti scegliendo, per ognuno, un offset che non si sovrappone.

    [B] con i gruppi appaiati (FedAvg/FedProx sono identici per costruzione) le
    etichette finiscono l'una sull'altra e diventano illeggibili. Qui ogni
    etichetta prova una lista di posizioni in ordine di preferenza e prende la
    prima libera; se nessuna lo e', prende quella con la sovrapposizione
    minore. Le zone occupate sono le etichette gia' poste PIU' i marker.

    items: lista di (x, y, testo, colore).
    """
    if not items:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # zone da evitare: i marker dei punti
    taken = []
    for x, y, _, _ in items:
        px, py = ax.transData.transform((x, y))
        taken.append(Bbox.from_bounds(px - 7, py - 7, 14, 14))

    cands = [(9, 6, "left", "bottom"), (9, -6, "left", "top"),
             (-9, 6, "right", "bottom"), (-9, -6, "right", "top"),
             (0, 13, "center", "bottom"), (0, -13, "center", "top"),
             (9, 20, "left", "bottom"), (9, -20, "left", "top"),
             (-9, 20, "right", "bottom"), (-9, -20, "right", "top"),
             (0, 26, "center", "bottom"), (0, -26, "center", "top"),
             (9, 34, "left", "bottom"), (-9, 34, "right", "bottom"),
             (9, -34, "left", "top"), (-9, -34, "right", "top")]

    # i punti piu' esterni scelgono per primi: hanno meno spazio libero attorno
    xs = [i[0] for i in items]
    cx = sum(xs) / len(xs)
    order = sorted(range(len(items)), key=lambda i: -abs(items[i][0] - cx))

    for i in order:
        x, y, text, color = items[i]
        ann = ax.annotate(text, (x, y), textcoords="offset points",
                          xytext=(9, 6), fontsize=8, color=color)
        best, best_ov = None, None
        for dx, dy, ha, va in cands:
            ann.set_position((dx, dy))
            ann.set_ha(ha)
            ann.set_va(va)
            bb = ann.get_window_extent(renderer=renderer).expanded(1.06, 1.35)
            ov = 0.0
            for t in taken:
                ix = max(0.0, min(bb.x1, t.x1) - max(bb.x0, t.x0))
                iy = max(0.0, min(bb.y1, t.y1) - max(bb.y0, t.y0))
                ov += ix * iy
            # penalizza le etichette che escono dagli assi
            ab = ax.get_window_extent(renderer=renderer)
            if bb.x0 < ab.x0 or bb.x1 > ab.x1 or bb.y0 < ab.y0 or bb.y1 > ab.y1:
                ov += 1e4
            if ov <= pad:
                best, best_ov = (dx, dy, ha, va), ov
                break
            if best_ov is None or ov < best_ov:
                best, best_ov = (dx, dy, ha, va), ov
        dx, dy, ha, va = best
        ann.set_position((dx, dy))
        ann.set_ha(ha)
        ann.set_va(va)
        taken.append(ann.get_window_extent(renderer=renderer).expanded(1.06, 1.35))


def fig_accounting(results_dir, out, beta):
    """Effetto della batteria dei device: una freccia per coppia.

    [B] la coda della freccia e' la run con batteria LINEARE, la punta la
    stessa run con batteria da DATASHEET (nm). La componente orizzontale dice
    quanti device in piu' o in meno si esauriscono con la batteria realistica,
    quella verticale quanto cambia l'accuratezza. Per SAGE la coppia cambia
    anche l'algoritmo (paper -> ablazione).
    """
    by_lab = _rows_by_beta(results_dir, beta)
    pt = {}
    for lab, runs in by_lab.items():
        vals = []
        for rows in runs:
            ar = _acc_rows(rows)
            if ar:
                vals.append((float(rows[-1]["n_soc_zero"]),
                             float(np.mean([x[1] for x in ar[-TAIL:]]))))
        if vals:
            pt[lab] = (float(np.mean([v[0] for v in vals])),
                       float(np.mean([v[1] for v in vals])))

    cand = PAIRS_GROUPED + PAIRS if GROUPED else PAIRS
    pairs = [(a, b) for a, b in cand if a in pt and b in pt]
    if not pairs:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.8))
    labels = []
    if "fedavg" in pt:
        ax.axhline(pt["fedavg"][1], color="#888888", ls=":", lw=1.2)
        ax.annotate("FedAvg accuracy", (ax.get_xlim()[1], pt["fedavg"][1]),
                    ha="right", va="bottom", fontsize=7.5, color="#888888")
        ax.plot(*pt["fedavg"], marker="s", ms=9, color="#444444")
        labels.append((pt["fedavg"][0], pt["fedavg"][1],
                       f"FedAvg ({pt['fedavg'][0]:.0f}\u2020)", "#444444"))

    heads = set()
    for lin, peu in pairs:
        c = COLOR.get(peu, "#333333")
        x0, y0 = pt[lin]
        x1, y1 = pt[peu]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=c,
                                    shrinkA=6, shrinkB=6, alpha=0.9))
        ax.plot(x0, y0, marker="o", ms=7, mfc="white", mec=c, mew=1.8)
        ax.plot(x1, y1, marker="o", ms=8, color=c)
        # [B] se piu' code puntassero alla stessa testa, la testa si etichetta
        # una volta sola
        labels.append((x0, y0, LABEL.get(lin, lin), COLOR.get(lin, c)))
        if peu not in heads:
            heads.add(peu)
            labels.append((x1, y1, LABEL.get(peu, peu).replace(" (SoC)", ""), c))

    ax.set_xlabel("Depleted clients at end of run (out of 30)")
    ax.set_ylabel(f"Test accuracy (mean of last {TAIL} rounds)")
    ax.set_title(f"Effect of the device battery model — β = {beta_str(beta)}\n"
                 "arrow tail = linear battery, head = datasheet battery")
    ax.grid(alpha=0.3)
    ax.margins(0.16)
    _place_labels(ax, labels)
    if beta == ALL:
        ax.text(0.01, -0.16, POOLED_NOTE, transform=ax.transAxes, fontsize=7.5,
                va="top", color="#555555")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True

def smooth(y, w=11):
    if len(y) < w:
        return np.asarray(y, dtype=float)
    k = np.ones(w) / w
    pad = w // 2
    yp = np.pad(np.asarray(y, dtype=float), pad, mode="edge")
    return np.convolve(yp, k, mode="valid")[: len(y)]


def load_curve(path):
    """-> beta -> algorithm -> seed -> [(round, energy, acc)]"""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path) as fh:
        rd = csv.DictReader(fh)
        has_beta = "beta" in (rd.fieldnames or [])
        for r in rd:
            beta = r["beta"] if has_beta else "n/d"
            algo = r["algorithm"]
            # [B] la chiave e' (variante, seed), non il solo seed: dopo il
            # raggruppamento piu' varianti condividono lo stesso seed e si
            # sovrascriverebbero a vicenda. Ogni variante entra nel gruppo
            # come una "ripetizione" in piu'.
            data[beta][group_of(algo)][f"{algo}#{r['seed']}"].append(
                (int(r["round"]), float(r["energy_wh"]), float(r["accuracy"]))
            )
    # [B] pseudo-beta "all": ogni (beta, variante, seed) entra come una
    # ripetizione in piu' del gruppo. La chiave include il beta, altrimenti
    # le curve di beta diversi si sovrascriverebbero.
    betas = [b for b in data if b != ALL]
    for b in betas:
        for a in data[b]:
            for s, v in data[b][a].items():
                data[ALL][a][f"b{b}#{s}"] = list(v)
    for b in data:
        for a in data[b]:
            for s in data[b][a]:
                data[b][a][s].sort()
    return data


def order_of(d):
    base = ORDER_GROUPED + ORDER if GROUPED else ORDER
    seen, out = set(), []
    for a in base:
        if a in d and a not in seen:
            out.append(a)
            seen.add(a)
    return out + [a for a in sorted(d) if a not in seen]


def _median_band(seeds, smooth_w):
    n = min(len(v) for v in seeds)
    E = np.median([[x[1] for x in v[:n]] for v in seeds], axis=0)
    R = [x[0] for x in seeds[0][:n]]
    A = np.array([smooth([x[2] for x in v[:n]], smooth_w) for v in seeds])
    return R, E, np.median(A, 0), np.percentile(A, 25, 0), np.percentile(A, 75, 0)


def _rows_by_beta(results_dir, beta):
    """label -> lista di righe (una per seed) dei CSV per-round del beta."""
    want = btag(beta)
    out = defaultdict(list)
    pooled = beta == ALL
    for path in glob.glob(os.path.join(results_dir, "*_seed*.csv")):
        name = os.path.basename(path)
        m = PAT_BETA.match(name)
        if m:
            lab, tag = m.group(1), m.group(2)
        else:
            m = PAT_PLAIN.match(name)
            if not m:
                continue
            lab, tag = m.group(1), "na"
        if not pooled and tag != want:
            continue
        with open(path) as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("mean_soc")]
        if rows:
            out[group_of(lab)].append(rows)
    return out


def _acc_rows(rows):
    """(round, accuracy, mean_soc, n_soc_zero) per le righe con accuracy."""
    out = []
    for r in rows:
        if not r.get("accuracy"):
            continue
        try:
            out.append((int(r["round"]), float(r["accuracy"]),
                        float(r["mean_soc"]), float(r["n_soc_zero"])))
        except (TypeError, ValueError):
            continue
    return out


def fig_energy_accuracy(d, out, beta, smooth_w):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ns = 0
    for a in order_of(d):
        seeds = list(d[a].values())
        ns = max(ns, len(seeds))
        _, E, med, lo, hi = _median_band(seeds, smooth_w)
        ax.plot(E, med, label=LABEL.get(a, a), color=COLOR.get(a),
                linestyle=STYLE.get(a, "-"), lw=1.8)
        ax.fill_between(E, lo, hi, color=COLOR.get(a), alpha=0.10, lw=0)
    ax.set_xlabel("Cumulative energy drawn from batteries [Wh]")
    ax.set_ylabel("Test accuracy (moving average)")
    ax.set_title(f"Energy–accuracy trade-off — β = {beta_str(beta)} "
                 f"(median over {ns} seeds, IQR band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_accuracy_rounds(d, out, beta, smooth_w):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ns = 0
    for a in order_of(d):
        seeds = list(d[a].values())
        ns = max(ns, len(seeds))
        R, _, med, lo, hi = _median_band(seeds, smooth_w)
        ax.plot(R, med, label=LABEL.get(a, a), color=COLOR.get(a),
                linestyle=STYLE.get(a, "-"), lw=1.8)
        ax.fill_between(R, lo, hi, color=COLOR.get(a), alpha=0.10, lw=0)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Test accuracy (moving average)")
    ax.set_title(f"Accuracy per round — β = {beta_str(beta)} "
                 f"(median over {ns} seeds, IQR band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_pareto(d, out, beta):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # [B] nella serie pooled le barre sarebbero dispersione fra beta, non
    # incertezza: mostrarle inviterebbe a leggerle come errore. Solo il punto.
    bars = beta != ALL
    ns = 0
    labels = []
    for a in order_of(d):
        seeds = list(d[a].values())
        ns = max(ns, len(seeds))
        E = [v[-1][1] for v in seeds]
        # media di coda, non massimo: a beta basso l'accuracy oscilla molto e
        # il massimo premia chi ha avuto un round fortunato
        A = [float(np.mean([x[2] for x in v[-TAIL:]])) for v in seeds]
        mx, my = float(np.mean(E)), float(np.mean(A))
        ax.errorbar(mx, my, xerr=np.std(E) if bars else None,
                    yerr=np.std(A) if bars else None,
                    fmt="o", ms=9, capsize=3 if bars else 0,
                    color=COLOR.get(a), label=LABEL.get(a, a))
        labels.append((mx, my, LABEL.get(a, a), COLOR.get(a, "#333333")))
    ax.set_xlabel("Total energy drawn from batteries [Wh]")
    ax.set_ylabel(f"Test accuracy (mean of last {TAIL} rounds)")
    ax.set_title(f"Energy / accuracy trade-off — β = {beta_str(beta)}\n"
                 + (f"(mean over {ns} seeds, bars = std. dev.; top-left is better)"
                    if bars else f"(mean over {ns} runs, top-left is better)"))
    ax.grid(alpha=0.3)
    ax.margins(0.14)
    _place_labels(ax, labels)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_soc(results_dir, out, beta):
    """SoC finale per algoritmo, dai CSV per-round del beta richiesto."""
    by_lab = _rows_by_beta(results_dir, beta)
    if not by_lab:
        return False
    stats = {
        lab: [(float(rows[-1]["mean_soc"]), float(rows[-1]["median_soc"]),
               float(rows[-1]["n_soc_zero"])) for rows in runs]
        for lab, runs in by_lab.items()
    }
    labs = order_of(stats)
    x = np.arange(len(labs))
    mean = [np.mean([v[0] for v in stats[a]]) for a in labs]
    med = [np.mean([v[1] for v in stats[a]]) for a in labs]
    dead = [np.mean([v[2] for v in stats[a]]) for a in labs]
    err = [np.std([v[0] for v in stats[a]]) for a in labs]

    bars = beta != ALL
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.bar(x - 0.2, mean, 0.4, yerr=err if bars else None,
            capsize=3 if bars else 0, label="Mean SoC", color="#1f77b4")
    ax1.bar(x + 0.2, med, 0.4, label="Median SoC", color="#aec7e8")
    ax1.set_ylabel("State of charge at end of run")
    ax1.set_title(f"Final state of charge — β = {beta_str(beta)} (dead clients counted as 0)")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=9)
    ax2.bar(x, dead, 0.5, color="#d62728")
    ax2.set_ylabel("Depleted clients")
    ax2.set_title("Devices depleted by end of run")
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABEL.get(a, a) for a in labs], rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def _scatter_soc_acc(stats, out, beta, xlabel, ylabel, title, note=None):
    """Scatter comune a fig5/fig6/fig7.

    stats: label -> lista di (soc, acc, morti, annot) una per seed.

    [B] nella serie pooled le barre d'errore sono soppresse: sarebbero la
    dispersione FRA BETA, cioe' l'effetto del regime, e disegnarle come barre
    invita a leggerle come incertezza. Resta il solo punto medio.
    """
    if not stats:
        return False
    bars = beta != ALL
    fig, ax = plt.subplots(figsize=(9, 5.8))
    labels = []
    for a in order_of(stats):
        S = [v[0] for v in stats[a]]
        A = [v[1] for v in stats[a]]
        D = float(np.mean([v[2] for v in stats[a]]))
        annot = stats[a][0][3]
        mx, my = float(np.mean(S)), float(np.mean(A))
        ax.errorbar(mx, my, xerr=np.std(S) if bars else None,
                    yerr=np.std(A) if bars else None,
                    fmt="o", ms=6 + 2.5 * np.sqrt(D), capsize=3 if bars else 0,
                    alpha=0.85, color=COLOR.get(a), label=LABEL.get(a, a))
        tag = LABEL.get(a, a)
        if annot:
            tag += f" {annot}"
        if D > 0:
            tag += f" ({D:.0f}\u2020)"
        labels.append((mx, my, tag, COLOR.get(a, "#333333")))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.margins(0.16)
    _place_labels(ax, labels)
    if note:
        ax.text(0.01, -0.16, note, transform=ax.transAxes, fontsize=7.5,
                va="top", color="#555555")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_pareto_soc(results_dir, out, beta):
    """Accuracy di coda vs SoC medio A FINE RUN."""
    by_lab = _rows_by_beta(results_dir, beta)
    stats = {}
    for lab, runs in by_lab.items():
        pts = []
        for rows in runs:
            ar = _acc_rows(rows)
            if not ar:
                continue
            pts.append((float(rows[-1]["mean_soc"]),
                        float(np.mean([x[1] for x in ar[-TAIL:]])),
                        float(rows[-1]["n_soc_zero"]), ""))
        if pts:
            stats[lab] = pts
    return _scatter_soc_acc(
        stats, out, beta,
        "Mean state of charge at end of run (dead clients counted as 0)",
        f"Test accuracy (mean of last {TAIL} rounds)",
        f"Survival / accuracy at end of run — β = {beta_str(beta)}\n"
        "larger marker = more depleted clients (\u2020); top-right is better")


def fig_peak_soc(results_dir, out, beta):
    """Accuracy MASSIMA vs SoC medio NEL ROUND DEL PICCO.

    [B] il momento di misura differisce fra algoritmi: chi raggiunge il picco
    presto appare con SoC alto per il solo fatto di aver misurato prima. Il
    round del picco e' annotato accanto a ogni punto proprio per questo.
    A beta basso l'accuracy oscilla di ~0.1 fra round consecutivi, quindi il
    massimo e' in parte il round piu' fortunato.
    """
    by_lab = _rows_by_beta(results_dir, beta)
    stats = {}
    for lab, runs in by_lab.items():
        pts, peak_rounds = [], []
        for rows in runs:
            ar = _acc_rows(rows)
            if not ar:
                continue
            rnd, acc, soc, dead = max(ar, key=lambda x: x[1])
            pts.append((soc, acc, dead, ""))
            peak_rounds.append(rnd)
        if pts:
            r_mean = int(np.mean(peak_rounds))
            stats[lab] = [(s, a, d, f"@r{r_mean}") for (s, a, d, _) in pts]
    return _scatter_soc_acc(
        stats, out, beta,
        "Mean state of charge at the peak round (dead clients counted as 0)",
        "Peak test accuracy",
        f"Survival / peak accuracy — β = {beta_str(beta)}\n"
        "larger marker = depleted clients (\u2020); top-right is better",
        note="Caution: the peak occurs at a different round for each method "
             "(annotated @r); a method peaking early is measured while its "
             "batteries are still full.")


def fig_threshold_soc(results_dir, out, beta, threshold=THRESHOLD):
    """SoC medio nel round in cui si raggiunge una SOGLIA comune.

    [B] questo e' il confronto equo: tutti misurati allo stesso risultato di
    apprendimento. Chi non raggiunge la soglia non compare, ed e' un dato.
    """
    by_lab = _rows_by_beta(results_dir, beta)
    stats, missing = {}, []
    for lab, runs in by_lab.items():
        pts, hit_rounds = [], []
        for rows in runs:
            ar = _acc_rows(rows)
            hit = next((x for x in ar if x[1] >= threshold), None)
            if hit is None:
                continue
            rnd, acc, soc, dead = hit
            pts.append((soc, acc, dead, ""))
            hit_rounds.append(rnd)
        if pts:
            r_mean = int(np.mean(hit_rounds))
            stats[lab] = [(s, a, d, f"@r{r_mean}") for (s, a, d, _) in pts]
        else:
            missing.append(LABEL.get(lab, lab))
    note = (f"Measured at the first round reaching accuracy {threshold:.2f} "
            "(round annotated @r), so every method is compared at the same "
            "learning outcome.")
    if missing:
        note += "\nNever reached the threshold: " + ", ".join(sorted(missing)) + "."
    return _scatter_soc_acc(
        stats, out, beta,
        f"Mean state of charge when accuracy {threshold:.2f} is first reached",
        f"Test accuracy at that round (\u2265 {threshold:.2f})",
        f"Survival at a common accuracy target — β = {beta_str(beta)}\n"
        "larger marker = depleted clients (\u2020); right is better",
        note=note)


def _per_round_group(runs, key):
    """Media sulle run del gruppo, round per round, di una colonna.

    [B] le varianti terminano a round diversi: a ogni round si media sulle
    sole run che sono arrivate fin li'. Dove una variante si esaurisce la
    curva puo' avere un gradino, ed e' un'informazione, non un artefatto.
    Ritorna (round, media, n_run_vive).
    """
    series = []
    for rows in runs:
        d = {}
        for r in rows:
            try:
                d[int(r["round"])] = float(r[key])
            except (TypeError, ValueError, KeyError):
                continue
        if d:
            series.append(d)
    if not series:
        return [], [], []
    rounds = sorted({x for s in series for x in s})
    mean, alive = [], []
    for rr in rounds:
        vals = [s[rr] for s in series if rr in s]
        mean.append(float(np.mean(vals)))
        alive.append(len(vals))
    return rounds, mean, alive


def fig_reserve(results_dir, out, beta, reserve=0.20):
    """SoC del client piu' scarico e morti cumulati, round per round.

    Mostra se la riserva tiene e quando i device iniziano a morire, per ogni
    algoritmo e batteria (tratteggio lineare, punteggiato Peukert, continuo
    datasheet).
    """
    by_lab = _rows_by_beta(results_dir, beta)
    if not by_lab:
        return False
    labs = order_of(by_lab)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    for a in labs:
        R, mn, _ = _per_round_group(by_lab[a], "min_soc")
        if R:
            ax1.plot(R, mn, color=COLOR.get(a), linestyle=STYLE.get(a, "-"),
                     lw=1.9, label=LABEL.get(a, a))
        R, dd, _ = _per_round_group(by_lab[a], "n_soc_zero")
        if R:
            ax2.plot(R, dd, color=COLOR.get(a), linestyle=STYLE.get(a, "-"),
                     lw=1.9, label=LABEL.get(a, a))

    ax1.axhline(reserve, color="k", ls=":", lw=1.2)
    ax1.annotate("reserve threshold", (1.0, reserve), xycoords=("axes fraction",
                 "data"), ha="right", va="bottom", fontsize=8, color="#555555")
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("State of charge of the most depleted client")
    ax1.set_title("Does the reserve hold?")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("Communication round")
    ax2.set_ylabel("Depleted clients (cumulative)")
    ax2.set_title("When devices start dying")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Battery reserve and device depletion \u2014 \u03b2 = {beta}\n"
                 "device battery: dashed = linear, dotted = Peukert, "
                 "solid = datasheet", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dir", default="results")
    ap.add_argument("--smooth", type=int, default=11)
    ap.add_argument("--beta", default=None,
                    help="grafica un solo beta; 'all' per la sola serie aggregata")
    ap.add_argument("--no-pooled", action="store_true",
                    help="non produrre la serie aggregata sui beta (_ball)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="soglia di accuracy per fig7")
    ap.add_argument("--reserve", type=float, default=0.20,
                    help="soglia di riserva del SoC, per fig9")
    ap.add_argument("--no-group", action="store_true",
                    help="non raggruppare le varianti ESCS")
    args = ap.parse_args()

    global GROUPED
    GROUPED = not args.no_group

    csv_path = args.csv or os.path.join(args.dir, "energy_accuracy_curve.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"manca {csv_path}: esegui prima analyze_rounds.py")
    data = load_curve(csv_path)
    if args.no_pooled:
        data.pop(ALL, None)

    # ordine: prima i beta singoli, poi la serie aggregata
    for beta in sorted(b for b in data if b != ALL) + ([ALL] if ALL in data else []):
        if args.beta and beta != args.beta:
            continue
        t = btag(beta)
        f1 = os.path.join(args.dir, f"fig1_energy_accuracy_b{t}.png")
        f2 = os.path.join(args.dir, f"fig2_accuracy_rounds_b{t}.png")
        f3 = os.path.join(args.dir, f"fig3_pareto_b{t}.png")
        f4 = os.path.join(args.dir, f"fig4_soc_b{t}.png")
        f5 = os.path.join(args.dir, f"fig5_pareto_soc_b{t}.png")
        f6 = os.path.join(args.dir, f"fig6_peak_soc_b{t}.png")
        f7 = os.path.join(args.dir, f"fig7_thr_soc_b{t}.png")
        f8 = os.path.join(args.dir, f"fig8_accounting_b{t}.png")
        f9 = os.path.join(args.dir, f"fig9_reserve_b{t}.png")
        fig_energy_accuracy(data[beta], f1, beta, args.smooth)
        fig_accuracy_rounds(data[beta], f2, beta, args.smooth)
        fig_pareto(data[beta], f3, beta)
        print(f"β={beta}:\n  {f1}\n  {f2}\n  {f3}")
        if fig_soc(args.dir, f4, beta):
            print(f"  {f4}")
        if fig_pareto_soc(args.dir, f5, beta):
            print(f"  {f5}")
        if fig_peak_soc(args.dir, f6, beta):
            print(f"  {f6}")
        if fig_threshold_soc(args.dir, f7, beta, args.threshold):
            print(f"  {f7}")
        if fig_accounting(args.dir, f8, beta):
            print(f"  {f8}")
        if fig_reserve(args.dir, f9, beta, args.reserve):
            print(f"  {f9}")


if __name__ == "__main__":
    main()