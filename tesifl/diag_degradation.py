#!/usr/bin/env python3
"""Diagnostica della degradazione post-picco.

    python diag_degradation.py                    # tutte le celle
    python diag_degradation.py --dir results_c200_n30

LA DOMANDA. Ogni algoritmo raggiunge un picco e poi cala. Due spiegazioni
incompatibili:
  (a) i client si esauriscono, il pool si restringe e si allontana dalla
      distribuzione globale, quindi il modello peggiora;
  (b) e' ottimizzazione: learning rate costante, 5 epoche locali su dati
      non-IID, client drift -- il calo ci sarebbe anche senza morti.

IL TEST. Le run con ZERO morti separano i due casi. Se calano anche quelle,
(a) e' esclusa come causa unica. Lo script le isola e le confronta.

COSA GUARDA, per ogni run:
  peak       round e accuratezza del massimo
  drop       picco meno media degli ultimi 20 round
  morti@peak quanti client erano gia' esauriti al momento del picco
  loss       se la loss globale sale mentre l'accuratezza scende, e' vera
             degradazione; se sale mentre l'accuratezza tiene, e' solo
             perdita di calibrazione e non va confusa con la prima.
"""
from __future__ import annotations
import argparse, csv, glob, os, re
import numpy as np

TAIL = 20
PAT = re.compile(r"^(.+)_b([0-9]+)_seed([0-9]+)\.csv$")


def rows_of(path):
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if r.get("accuracy")]


def f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def analyse(path):
    rows = rows_of(path)
    if len(rows) < TAIL + 5:
        return None
    acc = [f(r, "accuracy") for r in rows]
    i = int(np.argmax(acc))
    tail = float(np.mean(acc[-TAIL:]))
    dead_peak = f(rows[i], "n_soc_zero") or 0.0
    dead_end = f(rows[-1], "n_soc_zero") or 0.0
    l_peak, l_end = f(rows[i], "loss"), f(rows[-1], "loss")
    return {
        "rounds": len(rows), "peak_r": int(rows[i]["round"]), "peak": acc[i],
        "tail": tail, "drop": acc[i] - tail,
        "dead_peak": dead_peak, "dead_end": dead_end,
        "dloss": (l_end - l_peak) if (l_peak is not None and l_end is not None) else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="una cella; se omesso, tutte")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()

    dirs = [a.dir] if a.dir else sorted(glob.glob(os.path.join(a.root, "results_c*_n*")))
    allrows = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*_b*_seed*.csv"))):
            r = analyse(p)
            if r:
                r["cell"] = os.path.basename(d)
                r["run"] = os.path.basename(p).replace(".csv", "")
                allrows.append(r)
    if not allrows:
        print("nessun CSV")
        return

    zero = [r for r in allrows if r["dead_end"] == 0]
    some = [r for r in allrows if r["dead_end"] > 0]

    print(f"{len(allrows)} run analizzate\n")
    print("=" * 78)
    print("IL TEST: la degradazione c'e' anche senza morti?")
    print("=" * 78)
    for name, grp in (("run con ZERO morti", zero), ("run con morti", some)):
        if not grp:
            continue
        d = np.array([r["drop"] for r in grp])
        print(f"  {name:<22} n={len(grp):3d}   calo medio {d.mean():+.4f}   "
              f"mediano {np.median(d):+.4f}   max {d.max():+.4f}")
        big = sum(1 for x in d if x > 0.03)
        print(f"  {'':<22}       cali oltre 3 punti: {big}/{len(grp)}")
    print()
    if zero:
        dz = np.array([r["drop"] for r in zero])
        if np.median(dz) > 0.02:
            print("  -> LA DEGRADAZIONE C'E' ANCHE SENZA MORTI.")
            print("     L'esaurimento dei client non e' la causa (unica).")
            print("     Sospetti nell'ordine: learning rate costante su 250 round,")
            print("     5 epoche locali su dati non-IID (client drift), assenza di")
            print("     early stopping sul modello globale.")
        else:
            print("  -> senza morti il calo e' trascurabile: la degradazione segue")
            print("     l'esaurimento del pool, ed e' un effetto del regime, non")
            print("     dell'ottimizzazione.")
    print()

    # --- correlazione fra morti e calo
    if some and zero:
        x = np.array([r["dead_end"] for r in allrows], dtype=float)
        y = np.array([r["drop"] for r in allrows], dtype=float)
        if x.std() > 0:
            print(f"  correlazione(morti a fine run, calo) = {np.corrcoef(x, y)[0,1]:+.3f}")
            print("  (vicina a 0 = il calo non dipende dai morti)")
    print()

    # --- le peggiori, per capire dove guardare
    print("=" * 78)
    print("LE 12 RUN CON IL CALO MAGGIORE")
    print("=" * 78)
    print(f"{'cella':<18}{'run':<26}{'peak':>7}{'@r':>5}{'tail':>7}"
          f"{'calo':>7}{'morti':>7}{'dloss':>8}")
    for r in sorted(allrows, key=lambda r: -r["drop"])[:12]:
        print(f"{r['cell']:<18}{r['run'][:25]:<26}{r['peak']:7.3f}{r['peak_r']:5d}"
              f"{r['tail']:7.3f}{r['drop']:7.3f}{r['dead_end']:7.0f}{r['dloss']:8.2f}")
    print()
    print("dloss = loss globale a fine run meno loss al picco.")
    print("  loss che sale + accuratezza che scende -> degradazione vera")
    print("  loss che sale + accuratezza stabile    -> solo calibrazione")


if __name__ == "__main__":
    main()