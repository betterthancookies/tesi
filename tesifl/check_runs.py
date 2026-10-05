#!/usr/bin/env python3
"""Diagnostica dei CSV per-round: quanti round hanno davvero addestrato e
quanto costano rispetto a quello che il modello fisico prevede.

    python check_runs.py                 # tutti i CSV in results/
    python check_runs.py --dir results   --beta 05
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path


def analyse(path: Path) -> dict:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}

    def f(row, key):
        v = row.get(key, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    deltas, flat, prev = [], 0, None
    for row in rows:
        e = f(row, "total_energy_wh")
        if e is None:
            continue
        if prev is None:
            deltas.append(e)      # il round 1 e' esso stesso un incremento
        else:
            d = e - prev
            if d > 1e-12:
                deltas.append(d)
            else:
                flat += 1
        prev = e

    # round identici al precedente in TUTTE le colonne: nessun training
    frozen = 0
    for a, b in zip(rows, rows[1:]):
        if all(a.get(k) == b.get(k) for k in a if k != "round"):
            frozen += 1

    acc = [f(r, "accuracy") for r in rows if f(r, "accuracy") is not None]
    return {
        "rounds": len(rows),
        "charged": len(deltas),
        "flat": flat,
        "frozen": frozen,
        "med_mwh": st.median(deltas) * 1000 if deltas else 0.0,
        "min_mwh": min(deltas) * 1000 if deltas else 0.0,
        "max_mwh": max(deltas) * 1000 if deltas else 0.0,
        "total_wh": prev or 0.0,
        "acc_tail": st.mean(acc[-20:]) if acc else 0.0,
        "has_lin": "energy_lin_wh" in rows[0],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--beta", default="")
    ap.add_argument("--k", type=int, default=6, help="client per round attesi")
    args = ap.parse_args()

    pat = f"*{args.beta}*seed*.csv" if args.beta else "*seed*.csv"
    files = sorted(Path(args.dir).glob(pat))
    if not files:
        print(f"nessun CSV in {args.dir}")
        return

    print(f"{'file':<34}{'round':>6}{'addeb':>7}{'piatti':>7}{'gelati':>7}"
          f"{'mWh/round':>11}{'min':>8}{'max':>8}{'Wh tot':>9}{'lin':>5}")
    print("-" * 102)
    for p in files:
        a = analyse(p)
        if not a:
            continue
        flag = ""
        if a["charged"] < a["rounds"] * 0.98:
            flag += " <- round senza addebito"
        if a["frozen"]:
            flag += " <- round identici al precedente"
        print(f"{p.name[:33]:<34}{a['rounds']:>6}{a['charged']:>7}{a['flat']:>7}"
              f"{a['frozen']:>7}{a['med_mwh']:>11.1f}{a['min_mwh']:>8.1f}"
              f"{a['max_mwh']:>8.1f}{a['total_wh']:>9.3f}"
              f"{'si' if a['has_lin'] else 'NO':>5}{flag}")

    print()
    print("mWh/round e' la MEDIANA dell'incremento di total_energy_wh.")
    print(f"Atteso con k={args.k}, E=5, B=128 e partizioni Dirichlet: circa "
          f"{args.k * 10:.0f}-{args.k * 13:.0f} mWh "
          f"({args.k} client x ~10-13 mWh, secondo la taglia della partizione).")
    print("Molto sotto  -> meno client del previsto, o meno epoche, o partizioni piccole.")
    print("'piatti' = round senza incremento di energia; 'gelati' = round identici")
    print("al precedente in ogni colonna, cioe' nessun training avvenuto.")
    print("'lin' = NO significa CSV prodotto prima della doppia contabilita'.")


if __name__ == "__main__":
    main()