#!/usr/bin/env python3
"""Analisi della griglia multi-seed x multi-beta dai CSV per-round.

Da ~/tesifl (dopo run_grid.sh):
    python analyze_rounds.py                    tutti i beta trovati
    python analyze_rounds.py --beta 0.5         solo uno
    python analyze_rounds.py --dir results_smoke

Legge results/<label>_b<btag>_seed<N>.csv e produce, PER OGNI BETA:
  - tabella riassuntiva per algoritmo (media +- dev.std sui seed)
  - distribuzione del SoC a fine run
  - round-to-accuracy alle soglie richieste, con l'energia spesa per arrivarci
  - confronti APPAIATI per seed contro un riferimento (default: fedavg)
e un unico CSV con la curva energia-accuratezza, con colonna beta.

Il confronto appaiato e' necessario perche' la varianza fra seed supera
quella fra algoritmi: medie con deviazione standard indipendenti
nasconderebbero differenze sistematiche.

[B] i confronti appaiati usano l'accuracy di CODA (media degli ultimi TAIL
round), non quella dell'ultimo round: a beta=0.1 l'accuracy oscilla di ~0.1
fra round consecutivi, quindi il valore finale e' quasi un'estrazione casuale
da quella oscillazione e i delta che ne derivano sono rumore.

I beta NON si mediano fra loro: sono regimi diversi, non ripetizioni.
"""
import argparse
import csv
import glob
import os
import re
import statistics as st
from collections import defaultdict

THRESHOLDS = (0.25, 0.30, 0.35, 0.40)
TAIL = 20          # round di coda su cui mediare l'accuracy
REF = "fedavg"
ORDER = ["fedavg", "fedprox", "sage", "sage_lin", "sage_peuk", "sage_soc",
         "sage_smart_lin", "sage_smart_peuk", "sage_smart",
         "escs_sd_paper", "escs_sp_paper", "escs_md_paper", "escs_mp_paper",
         "escs_sd_lin", "escs_sp_lin", "escs_md_lin", "escs_mp_lin",
         "escs_sd_peuk", "escs_sp_peuk", "escs_md_peuk", "escs_mp_peuk",
         "escs_sd", "escs_sp", "escs_md", "escs_mp"]

# results/<label>_b<btag>_seed<N>.csv ; il vecchio schema senza beta e' accettato
PAT_BETA = re.compile(r"^(.+)_b([0-9]+)_seed([0-9]+)\.csv$")
PAT_PLAIN = re.compile(r"^(.+)_seed([0-9]+)\.csv$")


def btag_to_beta(tag):
    """01 -> 0.1, 05 -> 0.5, 10 -> 1.0."""
    if tag == "na":
        return "n/d"
    return f"{tag[0]}.{tag[1:]}" if len(tag) > 1 else tag


def load(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            def f(k):
                v = r.get(k, "")
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
            rows.append({
                "round": int(r["round"]), "accuracy": f("accuracy"),
                "loss": f("loss"), "energy": f("total_energy_wh"),
                "mean_soc": f("mean_soc"), "median_soc": f("median_soc"),
                "min_soc": f("min_soc"), "max_soc": f("max_soc"),
                "n_soc_zero": f("n_soc_zero"), "n_failed": f("n_failed"),
                "n_recharging": f("n_recharging"),
            })
    return rows


def summarize(rows):
    accs = [r for r in rows if r["accuracy"] is not None]
    last = rows[-1]
    body = [r for r in rows if r["energy"] is not None] or [last]
    out = {
        "acc_final": accs[-1]["accuracy"] if accs else None,
        # [B] media di coda: e' questa che entra nei confronti appaiati
        "acc_tail": (st.mean([r["accuracy"] for r in accs[-TAIL:]])
                     if accs else None),
        "acc_max": max(r["accuracy"] for r in accs) if accs else None,
        "energy": body[-1]["energy"],
        "mean_soc": body[-1]["mean_soc"],
        "median_soc": body[-1]["median_soc"],
        "min_soc": body[-1]["min_soc"],
        "max_soc": body[-1]["max_soc"],
        "n_dead": body[-1]["n_soc_zero"],
        "rounds": last["round"],
    }
    # picco: accuracy massima e stato del mondo nel round in cui avviene
    if accs:
        peak = max(accs, key=lambda r: r["accuracy"])
        out["peak_round"] = peak["round"]
        out["peak_soc"] = peak["mean_soc"]
    else:
        out["peak_round"] = out["peak_soc"] = None

    for th in THRESHOLDS:
        hit = next((r for r in accs if r["accuracy"] >= th), None)
        out[f"r2a_{th}"] = hit["round"] if hit else None
        out[f"e2a_{th}"] = hit["energy"] if hit else None
        out[f"s2a_{th}"] = hit["mean_soc"] if hit else None
    return out


def agg(vals, fmt="{:.4f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/d"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(st.mean(vals))}±{fmt.format(st.stdev(vals))}"


def report(beta, data, ref, thr_focus=0.40):
    """data: label -> seed -> summary"""
    labels = [l for l in ORDER if l in data] + \
             [l for l in sorted(data) if l not in ORDER]

    print("\n" + "#" * 108)
    print(f"#  BETA = {beta}")
    print("#" * 108)

    print(f"{'algoritmo':<14}{'n':>2} {'acc coda':>15} {'acc max':>15} "
          f"{'energia Wh':>15} {'SoC medio':>15} {'morti':>9} {'round':>8}")
    print("-" * 108)
    for lab in labels:
        rs = list(data[lab].values())
        print(f"{lab:<14}{len(rs):>2} "
              f"{agg([r['acc_tail'] for r in rs]):>15} "
              f"{agg([r['acc_max'] for r in rs]):>15} "
              f"{agg([r['energy'] for r in rs]):>15} "
              f"{agg([r['mean_soc'] for r in rs]):>15} "
              f"{agg([r['n_dead'] for r in rs], '{:.1f}'):>9} "
              f"{agg([r['rounds'] for r in rs], '{:.0f}'):>8}")
    print(f"acc coda = media degli ultimi {TAIL} round. La colonna round "
          "smaschera le run terminate prima del previsto.")

    print("\nDISTRIBUZIONE DEL SoC A FINE RUN (morti inclusi come 0)")
    print(f"{'algoritmo':<14} {'medio':>15} {'mediano':>15} {'min':>15} {'max':>15}")
    print("-" * 108)
    for lab in labels:
        rs = list(data[lab].values())
        print(f"{lab:<14} {agg([r['mean_soc'] for r in rs]):>15} "
              f"{agg([r['median_soc'] for r in rs]):>15} "
              f"{agg([r['min_soc'] for r in rs]):>15} "
              f"{agg([r['max_soc'] for r in rs]):>15}")

    # ---- picco e soglia: dove si trova il mondo quando si misura ----
    print(f"\nPICCO E SOGLIA (SoC medio nel momento della misura)")
    print(f"{'algoritmo':<14} {'acc picco':>12} {'round':>7} {'SoC@picco':>12}"
          f" {'SoC@' + f'{thr_focus:.2f}':>12} {'round':>7}")
    print("-" * 108)
    for lab in labels:
        rs = list(data[lab].values())
        r2a = [r[f"r2a_{thr_focus}"] for r in rs if r[f"r2a_{thr_focus}"] is not None]
        s2a = [r[f"s2a_{thr_focus}"] for r in rs if r[f"s2a_{thr_focus}"] is not None]
        print(f"{lab:<14} {agg([r['acc_max'] for r in rs]):>12} "
              f"{agg([r['peak_round'] for r in rs], '{:.0f}'):>7} "
              f"{agg([r['peak_soc'] for r in rs]):>12} "
              f"{(agg(s2a) if s2a else 'mai'):>12} "
              f"{(agg(r2a, '{:.0f}') if r2a else '-'):>7}")
    print("[B] il picco cade a round diversi: chi lo raggiunge presto ha un "
          "SoC alto solo perche' misura prima. La colonna alla soglia comune "
          "e' il confronto equo.")

    print("\nROUND-TO-ACCURACY (round | energia Wh spesa per arrivarci)")
    hdr = f"{'algoritmo':<14}"
    for th in THRESHOLDS:
        hdr += f"{('acc>=' + str(th)):>22}"
    print(hdr)
    print("-" * 108)
    for lab in labels:
        rs = list(data[lab].values())
        line = f"{lab:<14}"
        for th in THRESHOLDS:
            r2a = [r[f"r2a_{th}"] for r in rs if r[f"r2a_{th}"] is not None]
            e2a = [r[f"e2a_{th}"] for r in rs if r[f"e2a_{th}"] is not None]
            if not r2a:
                line += f"{'mai':>22}"
            else:
                miss = len(rs) - len(r2a)
                tag = f"{st.mean(r2a):.0f}r {st.mean(e2a):.3f}Wh"
                if miss:
                    tag += f" ({miss}x mai)"
                line += f"{tag:>22}"
        print(line)

    if ref not in data:
        return
    print(f"\nCONFRONTI APPAIATI PER SEED (riferimento: {ref}, acc di coda)")
    print(f"{'algoritmo':<14}{'seed vinti':>12}{'Δ accuracy':>20}{'Δ energia %':>20}")
    print("-" * 108)
    for lab in labels:
        if lab == ref:
            continue
        seeds = sorted(set(data[lab]) & set(data[ref]))
        da, de = [], []
        for s in seeds:
            a, b = data[lab][s], data[ref][s]
            if a["acc_tail"] is not None and b["acc_tail"] is not None:
                da.append(a["acc_tail"] - b["acc_tail"])
            if a["energy"] and b["energy"]:
                de.append(100 * (1 - a["energy"] / b["energy"]))
        if not da:
            continue
        wins = sum(1 for x in da if x > 0)
        print(f"{lab:<14}{f'{wins}/{len(da)}':>12}"
              f"{agg(da, '{:+.4f}'):>20}{agg(de, '{:+.1f}'):>20}")
    print("Δ energia positivo = consuma MENO del riferimento.")
    print("Con n seed piccoli il test dei segni non e' significativo "
          "(3/3 -> p=0.125, 4/4 -> p=0.062, 5/5 -> p=0.031).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--beta", default=None, help="analizza un solo beta, es. 0.5")
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="soglia della tabella picco/soglia")
    args = ap.parse_args()

    # beta -> label -> seed -> summary
    data = defaultdict(lambda: defaultdict(dict))
    index = []   # (beta, label, seed, path) per la curva
    for path in sorted(glob.glob(os.path.join(args.dir, "*_seed*.csv"))):
        name = os.path.basename(path)
        m = PAT_BETA.match(name)
        if m:
            label, btag, seed = m.group(1), m.group(2), int(m.group(3))
        else:
            m = PAT_PLAIN.match(name)
            if not m:
                continue
            label, btag, seed = m.group(1), "na", int(m.group(2))
        beta = btag_to_beta(btag)
        if args.beta and beta != args.beta:
            continue
        try:
            data[beta][label][seed] = summarize(load(path))
            index.append((beta, label, seed, path))
        except (KeyError, IndexError, ValueError) as e:
            print(f"  [!] {path}: {e}")

    if not data:
        raise SystemExit(
            f"nessun CSV in {args.dir}/ (atteso <label>_b<btag>_seed<N>.csv)")

    for beta in sorted(data):
        report(beta, data[beta], args.ref, args.threshold)

    out = os.path.join(args.dir, "energy_accuracy_curve.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beta", "algorithm", "seed", "round", "energy_wh", "accuracy"])
        for beta, label, seed, path in sorted(index):
            for r in load(path):
                if r["accuracy"] is not None and r["energy"] is not None:
                    w.writerow([beta, label, seed, r["round"],
                                f"{r['energy']:.6f}", f"{r['accuracy']:.6f}"])
    print(f"\nCurva energia-accuratezza (per il grafico) in {out}")


if __name__ == "__main__":
    main()