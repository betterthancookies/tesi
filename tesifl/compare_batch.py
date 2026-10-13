#!/usr/bin/env python3
"""Confronto dello sweep sul batch (sweep_batch.sh): per ogni algoritmo e B,
accuracy finale contro batteria, media sui beta e beta per beta.

    python compare_batch.py                      # results_sweepB*/ qui
    python compare_batch.py --root .. --last 20

acc     = media dell'accuracy negli ultimi --last round valutati
SoC     = SoC medio finale su tutta la popolazione (morti inclusi col loro 0)
morti   = device con SoC 0 a fine run
Wh      = carica totale spesa dalle batterie
round   = round completati (meno di quelli previsti = run interrotta)
Il B migliore e' quello che sta piu' in alto a destra sul piano (acc, SoC):
piu' accuracy senza pagarla in batteria.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics as st
from pathlib import Path


def summarize(path: Path, last: int) -> dict | None:
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("mean_soc")]
    if not rows:
        return None

    def f(r, k):
        try:
            return float(r[k])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    acc = [f(r, "accuracy") for r in rows if r.get("accuracy")]
    end = rows[-1]
    return {"acc": st.mean(acc[-last:]) if acc else float("nan"),
            "soc": f(end, "mean_soc"), "dead": f(end, "n_soc_zero"),
            "wh": f(end, "total_energy_wh"), "rounds": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--last", type=int, default=20)
    args = ap.parse_args()

    # (algoritmo, B, beta) -> riassunto
    res: dict[tuple[str, int, str], dict] = {}
    for d in sorted(Path(args.root).glob("results_sweepB*")):
        m = re.fullmatch(r"results_sweepB(\d+)", d.name)
        if not m:
            continue
        b = int(m.group(1))
        for p in d.glob("*_b*_seed*.csv"):
            mm = re.fullmatch(r"(.+)_b(\d+)_seed(\d+)\.csv", p.name)
            s = summarize(p, args.last) if mm else None
            if s:
                res[(mm.group(1), b, mm.group(2))] = s
    if not res:
        print(f"nessun CSV in {args.root}/results_sweepB*/")
        return

    algos = sorted({k[0] for k in res})
    bs = sorted({k[1] for k in res})
    betas = sorted({k[2] for k in res})
    head = f"{'B':>5}{'acc':>8}{'SoC':>8}{'morti':>8}{'Wh':>8}{'run':>6}"
    for a in algos:
        print(f"\n=== {a}: media sui beta {betas} ===")
        print(head)
        for b in bs:
            xs = [res[(a, b, be)] for be in betas if (a, b, be) in res]
            if not xs:
                continue
            g = lambda k: st.mean(x[k] for x in xs)  # noqa: E731
            print(f"{b:>5}{g('acc'):8.3f}{g('soc'):8.3f}{g('dead'):8.1f}"
                  f"{g('wh'):8.1f}{len(xs):>6}")
        print("  beta per beta (acc / SoC / morti):")
        for be in betas:
            cells = [f"B={b}: " + (f"{res[(a, b, be)]['acc']:.3f} / "
                                   f"{res[(a, b, be)]['soc']:.3f} / "
                                   f"{res[(a, b, be)]['dead']:.0f}"
                                   if (a, b, be) in res else "manca")
                     for b in bs]
            print(f"    b{be}  " + "   ".join(cells))
        short = {res[k]["rounds"] for k in res if k[0] == a}
        if len(short) > 1:
            print(f"  ATTENZIONE: round completati diversi fra le run: {sorted(short)}")


if __name__ == "__main__":
    main()
