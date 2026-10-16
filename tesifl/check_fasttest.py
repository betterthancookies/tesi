#!/usr/bin/env python3
"""Controlla il CSV della run minuscola di `bash run.sh --fastTest`.

    python check_fasttest.py --csv results_fasttest/fedavg_b05_seed2026.csv \
        --rounds 3 --clients 30 [--label fedavg] [--seconds 140] [--warn "..."]

In ogni mondo la carica spesa dalla batteria e' la somma delle sue voci
(training + comunicazione + idle + controllo). Nel mondo LINEARE (etichette
senza suffisso _peuk/_nm, come fedavg) e' anche uguale all'energia del carico.
Gli intervalli sono larghi (circa 10x attorno ai valori della campagna
rechargeONdeviceON_ResNet20_batterie, beta 0.5, 3 round: accuracy 0.38,
training ~0.6 Wh/round, comunicazione ~0.05 Wh/round per 6 client): devono
scattare per un errore vero, non per una variazione della fisica.
Con un altro modello o dataset ([model] di experiment.toml) le soglie si
scalano: accuracy con il numero di classi, training con i MAC per campione e
la frazione di client, comunicazione con i MB del modello e i client per round.
Exit code 0 se tutto ok, 1 altrimenti.
"""
from __future__ import annotations

import argparse
import csv
import math
import tomllib
from pathlib import Path

COLS = ("round", "accuracy", "loss", "total_energy_wh", "energy_train_wh",
        "energy_comm_wh", "energy_idle_wh", "energy_ctrl_wh", "energy_lin_wh",
        "mean_soc", "min_soc", "max_soc", "n_soc_zero", "n_clients",
        "mean_workload")

# campagna di riferimento delle soglie: ResNet-20 su CIFAR-10, 6 client per round
REF_MACS, REF_PARAMS, REF_K_FRACTION, REF_K = 40551040, 269722, 0.2, 6
N_CLASSES = {"cifar10": 10, "cifar100": 100}


def scales(clients: int) -> tuple[str, int, float, float]:
    """(modello, classi, scala del training, scala della comunicazione)."""
    cfg = tomllib.load(open("experiment.toml", "rb"))
    name, ds = cfg["model"]["name"], cfg["model"]["dataset"]
    spec = cfg["model"]["specs"][f"{name}_{ds}"]
    kf = float(cfg["campaign"]["k_fraction"])
    k = max(1, round(kf * clients))
    # training per round ~ MAC x campioni dei selezionati (50k x k_fraction)
    s_train = max(1.0, spec["macs_fwd_per_sample"] / REF_MACS * kf / REF_K_FRACTION)
    # comunicazione per round ~ MB del modello x client per round
    s_comm = max(1.0, spec["n_params"] / REF_PARAMS * k / REF_K)
    return f"{name}/{ds}", N_CLASSES.get(ds, 10), s_train, s_comm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--clients", type=int, required=True)
    ap.add_argument("--label", default="fedavg")
    ap.add_argument("--seconds", type=int, default=0)
    ap.add_argument("--warn", default="", help="avvisi dai controlli d'ambiente")
    args = ap.parse_args()

    fails: list[str] = []
    warns: list[str] = [w for w in args.warn.split("|") if w]

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        if not ok:
            fails.append(name)

    print("\n=== controllo dei valori ===")
    path = Path(args.csv)
    check("CSV prodotto", path.is_file() and path.stat().st_size > 0, str(path))
    if fails:
        return verdict(fails, warns)

    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    missing = [c for c in COLS if rows and c not in rows[0]]
    check("colonne attese presenti", not missing, ", ".join(missing))
    try:
        rnds = [int(float(r["round"])) for r in rows]
    except (TypeError, ValueError):
        rnds = []
    check(f"{args.rounds} round registrati",
          rnds == list(range(1, args.rounds + 1)), f"trovati {len(rows)}")
    if fails:
        return verdict(fails, warns)

    def col(k: str) -> list[float]:
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out

    v = {k: col(k) for k in COLS}
    bad = [k for k in COLS if any(not math.isfinite(x) for x in v[k])]
    check("nessun valore vuoto o NaN", not bad, ", ".join(bad))
    if fails:
        return verdict(fails, warns)

    model, n_cls, s_train, s_comm = scales(args.clients)
    print(f"  (modello {model}: soglie training x{s_train:.1f}, comunicazione x{s_comm:.1f})")

    # --- modello: impara qualcosa ---------------------------------------
    acc = v["accuracy"]
    acc_min = 1.5 / n_cls
    check("accuracy fra 0 e 1", all(0.0 <= a <= 1.0 for a in acc))
    check(f"il modello impara (accuracy finale > {acc_min:.3f}, caso = {1 / n_cls:.2f})",
          acc[-1] > acc_min, f"{' -> '.join(f'{a:.3f}' for a in acc)}")
    check("loss finita e positiva", all(x > 0 for x in v["loss"]),
          f"{' -> '.join(f'{x:.3f}' for x in v['loss'])}")

    # --- ogni round ha addestrato davvero --------------------------------
    # [B] 09/09: con la GPU contesa ~20% dei round non addestro' nessuno, senza
    # errori: si vede solo come energia di training ferma da un round all'altro
    tr = [v["energy_train_wh"][0]] + [b - a for a, b in
                                      zip(v["energy_train_wh"], v["energy_train_wh"][1:])]
    cm = [v["energy_comm_wh"][0]] + [b - a for a, b in
                                     zip(v["energy_comm_wh"], v["energy_comm_wh"][1:])]
    check("training in ogni round (energia di training sempre in aumento)",
          all(d > 0 for d in tr), " ".join(f"{d:.3f}" for d in tr) + " Wh")
    t_lo, t_hi = 0.05 * s_train, 5.0 * s_train
    c_lo, c_hi = 0.001 * s_comm, 0.5 * s_comm
    check(f"training per round plausibile ({t_lo:.2g}-{t_hi:.2g} Wh)",
          all(t_lo <= d <= t_hi for d in tr))
    check(f"comunicazione per round plausibile ({c_lo:.2g}-{c_hi:.2g} Wh)",
          all(c_lo <= d <= c_hi for d in cm), " ".join(f"{d:.4f}" for d in cm) + " Wh")
    check("idle in aumento", all(b > a for a, b in
                                 zip(v["energy_idle_wh"], v["energy_idle_wh"][1:])))

    # --- contabilita' dell'energia ----------------------------------------
    parts = [t + c + i + k for t, c, i, k in zip(v["energy_train_wh"], v["energy_comm_wh"],
                                                  v["energy_idle_wh"], v["energy_ctrl_wh"])]
    err_sum = max(abs(p - e) / e for p, e in zip(parts, v["total_energy_wh"]))
    check("carica = training + comunicazione + idle + controllo", err_sum < 1e-3,
          f"errore relativo {err_sum:.1e}")
    if not args.label.endswith(("_peuk", "_nm")):
        err_lin = max(abs(t - e) / e for t, e in zip(v["total_energy_wh"], v["energy_lin_wh"]))
        check("batteria lineare: carica della batteria = energia del carico", err_lin < 1e-3,
              f"errore relativo {err_lin:.1e}")

    # --- SAGE-smart: B ed E scelti dai client --------------------------------
    if args.label.startswith("sage_smart") and "mean_batch" in rows[0]:
        mb = [x for x in (float(r["mean_batch"]) for r in rows if r.get("mean_batch"))
              if math.isfinite(x)]
        me = [x for x in (float(r["mean_epochs"]) for r in rows if r.get("mean_epochs"))
              if math.isfinite(x)]
        al = tomllib.load(open("experiment.toml", "rb"))["algo"]
        b_lo, b_hi = int(al["smart_batch_min"]), int(al["smart_batch_max"])
        check(f"B medio scelto dai client dentro l'envelope ({b_lo}-{b_hi})",
              bool(mb) and all(b_lo <= b <= b_hi for b in mb),
              "B " + " ".join(f"{b:.0f}" for b in mb)
              + ("  E " + " ".join(f"{e:.1f}" for e in me) if me else ""))
        # [B] FedNova amplifica di tau_eff/tau_i chi fa pochi passi: sopra ~8
        # un solo update domina (collasso del 14/10 con spread 15-16)
        sp = [x for x in (float(r["tau_spread"]) for r in rows if r.get("tau_spread"))
              if math.isfinite(x)]
        if sp:
            check("spread dei passi FedNova contenuto (max tau / min tau <= 8)",
                  max(sp) <= 8.0, " ".join(f"{x:.1f}" for x in sp))

    # --- broadcast dello stato (solo strategie che selezionano sullo stato) --
    if rows[0].get("bcast_targets") not in (None, ""):
        tg = [int(float(r["bcast_targets"])) for r in rows]
        rs = [int(float(r["bcast_resp"])) for r in rows]
        eb = [float(r["energy_bcast_wh"]) for r in rows]
        share = 1.0 - sum(rs) / max(1, sum(tg))
        check("broadcast: risposte <= richieste, energia in aumento",
              all(a <= b for a, b in zip(rs, tg)) and all(y >= x for x, y in zip(eb, eb[1:])),
              f"risposte {' '.join(map(str, rs))} su {' '.join(map(str, tg))}, "
              f"{eb[-1] * 1000:.1f} mWh")
        if not 0.0 <= share <= 0.4:
            warns.append(f"broadcast: non risponde il {share:.0%}, atteso ~10%")

    # --- batterie ----------------------------------------------------------
    soc_ok = all(0.0 <= lo <= m <= hi <= 1.0 for lo, m, hi in
                 zip(v["min_soc"], v["mean_soc"], v["max_soc"]))
    check("SoC ordinati e in [0, 1] (min <= media <= max)", soc_ok)
    check("SoC medio plausibile dopo pochi round (0.2-1)",
          0.2 <= v["mean_soc"][-1] <= 1.0, f"{v['mean_soc'][-1]:.3f}")
    check(f"{args.clients} client nel mondo", all(int(n) == args.clients for n in v["n_clients"]))
    check("workload medio in [0, 1]", all(0.0 <= w <= 1.0 for w in v["mean_workload"]))
    dead = int(v["n_soc_zero"][-1])
    check("device scarichi fra 0 e N", 0 <= dead <= args.clients, f"{dead}")
    if rows[0].get("n_soc_low") not in (None, ""):
        low = [int(float(r["n_soc_low"])) for r in rows]
        check("device sotto il 20% (colonna n_soc_low) >= device scarichi",
              all(lo >= int(z) for lo, z in zip(low, v["n_soc_zero"])),
              " ".join(str(x) for x in low))
    else:
        warns.append("manca la colonna n_soc_low (fig10): codice vecchio sul server?")
    if dead > 0:
        warns.append(f"{dead} device scarichi dopo {args.rounds} round: insolito, controlla")

    if args.seconds:
        print(f"\n  durata della run minuscola: {args.seconds} s "
              f"(compresi avvio di Ray e caricamento del dataset)")
    return verdict(fails, warns)


def verdict(fails: list[str], warns: list[str]) -> int:
    print("")
    for w in warns:
        print(f"  ATTENZIONE: {w}")
    if fails:
        print(f"=== PROBLEMI ({len(fails)}): NON procedere ===")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("=== VALORI CORRETTI, PROCEDERE ===")
    print("  campagna:  bash run.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
