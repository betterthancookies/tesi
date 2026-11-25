import csv
import os
import time
from pathlib import Path

_ACC_BY_ROUND: dict[int, tuple[float, float]] = {}

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from device.world_state import BATTERY_NAMES, build_world_state
from strategies.fedavg import PhysicalFedAvg
from strategies.fedprox import PhysicalFedProx
from strategies.sage import PhysicalSAGE
from strategies.sage_ablation import PhysicalSAGEAblation
from strategies.sage_smart import PhysicalSAGESmart
from strategies.escs import PhysicalESCS
from tesiFL.data.partition import load_centralized_dataset, num_classes, partition_sizes
from tesiFL.task import test
from tesiFL.data.model import build_model

app = ServerApp()

_MODEL_NAME = "resnet20"
_DATASET = "cifar10"


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    epochs: int = context.run_config["local-epochs"]
    batch_size: int = context.run_config["batch-size"]
    seed: int = context.run_config["seed"]
    algorithm: str = str(context.run_config["algorithm"]).lower()
    proximal_mu: float = context.run_config["proximal-mu"]
    global _MODEL_NAME, _DATASET
    _MODEL_NAME = str(context.run_config["model-name"])
    _DATASET = str(context.run_config["dataset"])
    from device.constants import MODEL_DATASET, MODEL_NAME
    if (_MODEL_NAME, _DATASET) != (MODEL_NAME, MODEL_DATASET):
        raise ValueError(
            f"run con {_MODEL_NAME}/{_DATASET} ma la fisica (experiment.toml, "
            f"[model]) e' per {MODEL_NAME}/{MODEL_DATASET}: cambia [model] li'")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global_model = build_model(_MODEL_NAME, num_classes(_DATASET))
    arrays = ArrayRecord(global_model.state_dict())

    n_clients = len(grid.get_node_ids())
    capacity_mah = float(context.run_config["battery-capacity-mah"])
    k_per_round = int(context.run_config["clients-per-round"])
    tiers_on = bool(context.run_config["tiers-enabled"])
    workload_on = bool(context.run_config["workload-enabled"])
    idle_on = bool(context.run_config["idle-enabled"])
    recharge_on = bool(context.run_config["recharge-enabled"])
    recharge_avail = bool(context.run_config["recharge-available"])
    battery = str(context.run_config["world-battery"])
    workload_scale = float(context.run_config["workload-scale"])
    world = build_world_state(n_clients=n_clients, seed=seed,
                              capacity_mah=capacity_mah,
                              tiers_enabled=tiers_on,
                              workload_enabled=workload_on,
                              idle_enabled=idle_on,
                              recharge_enabled=recharge_on,
                              recharge_available=recharge_avail,
                              battery=battery,
                              workload_scale=workload_scale)
    from collections import Counter
    tiers = Counter(world.tier(c) for c in range(n_clients))
    print(f"[world] batteria {BATTERY_NAMES[world.battery]} ({world.battery})")
    print(f"[world] {n_clients} SuperNode | k={k_per_round} "
          f"({k_per_round / max(n_clients, 1):.0%}) | "
          f"tier {'ON ' + str(dict(tiers)) if tiers_on else f'OFF ({capacity_mah:.0f} mAh)'}")
    print(f"[world] workload {'ON ' + str(world.workload.counts_by_profile()) if workload_on else 'OFF'}"
          f" x{workload_scale:g} (stato piu' pesante w={world.workload.max_utilization():.2f})"
          f" | idle {'ON' if idle_on else 'OFF'}"
          f" | ricarica {('ON, modalita ' + ('a: disponibile' if recharge_avail else 'b: unavailable')) if recharge_on else 'OFF'}")

    common = dict(
        world=world,
        epochs=epochs,
        batch_size=batch_size,
        k_per_round=k_per_round,
        selection_seed=seed,
        fraction_evaluate=fraction_evaluate,
    )
    bcast = dict(
        broadcast=bool(context.run_config["broadcast"]),
        broadcast_nonresp=float(context.run_config["broadcast-nonresp"]),
        broadcast_timeout_s=float(context.run_config["broadcast-timeout-s"]),
        broadcast_top_m=int(context.run_config["broadcast-top-m"]),
    )
    if bcast["broadcast"] and not algorithm.startswith(("fedavg", "fedprox", "sage_smart")):
        print(f"[strategy] broadcast dello stato: {bcast['broadcast_nonresp']:.0%} "
              f"non risponde, timeout {bcast['broadcast_timeout_s']} s, "
              f"{'tutti' if bcast['broadcast_top_m'] <= 0 else 'primi ' + str(bcast['broadcast_top_m'])}")
    if algorithm.startswith("escs"):
        suffix = algorithm.split("-")[-1] if "-" in algorithm else "sd"
        if len(suffix) != 2 or suffix[0] not in "sm" or suffix[1] not in "dp":
            raise ValueError(
                f"variante ESCS non valida: '{algorithm}' "
                "(usa escs-sd, escs-sp, escs-md, escs-mp)"
            )
        sizes = partition_sizes(
            num_partitions=n_clients,
            beta=float(context.run_config["beta"]),
            seed=seed,
            dataset=_DATASET,
        )
        strategy = PhysicalESCS(
            utility_mode=suffix[0], selection_mode=suffix[1],
            partition_sizes=sizes,
            first_round_all=bool(context.run_config["escs-first-round-all"]),
            cap_probabilistic=bool(context.run_config["escs-cap-probabilistic"]),
            min_battery=float(context.run_config["escs-min-battery"]),
            min_network_quality=float(context.run_config["escs-min-nq"]),
            **common, **bcast,
        )
        print(f"[strategy] PhysicalESCS ({strategy.variant})")
    elif algorithm == "sage":
        strategy = PhysicalSAGE(
            sage_a=float(context.run_config["sage-a"]),
            sage_b=float(context.run_config["sage-b"]),
            sage_c=float(context.run_config["sage-c"]),
            **common, **bcast,
        )
        print(
            f"[strategy] PhysicalSAGE (a={context.run_config['sage-a']}, "
            f"b={context.run_config['sage-b']}, c={context.run_config['sage-c']})"
        )
    elif algorithm == "sage_smart":
        sizes = partition_sizes(
            num_partitions=n_clients,
            beta=float(context.run_config["beta"]),
            seed=seed,
            dataset=_DATASET,
        )
        smart = dict(
            stale_weight=float(context.run_config["smart-stale-weight"]),
            epochs_min=int(context.run_config["smart-epochs-min"]),
            batch_min=int(context.run_config["smart-batch-min"]),
            batch_max=int(context.run_config["smart-batch-max"]),
        )
        ablation = str(context.run_config["smart-ablation"])
        ablations = {
            "none":      {},
            "nostale":   {"stale_weight": 0.0},
            "noest":     {"forecast": False},
            "noneg":     {"negotiation": False},
            "e5":        {"epochs_min": epochs, "epochs_cap_mult": 1},
            "b128":      {"batch_min": batch_size, "batch_max": batch_size},
            "nopartial": {"partial_work": False},
            "srvlin":    {"server_battery": "lin"},
        }
        parts = [p for p in ablation.split("+") if p and p != "none"]
        unknown = [p for p in parts if p not in ablations]
        if unknown:
            raise ValueError(f"smart-ablation sconosciuta: {'+'.join(unknown)!r} "
                             f"(usa {', '.join(ablations)}, uniti da '+')")
        for p in parts:
            smart.update(ablations[p])
        if smart.get("negotiation", True):
            bcast = dict(bcast, broadcast=False)
            print("[strategy] SAGE-smart: niente broadcast dello stato, "
                  f"{bcast['broadcast_nonresp']:.0%} di non risposte sulle proposte")
        elif bcast["broadcast"]:
            print(f"[strategy] SAGE-smart senza negoziazione: broadcast dello stato, "
                  f"{bcast['broadcast_nonresp']:.0%} non risponde, timeout "
                  f"{bcast['broadcast_timeout_s']} s")
        strategy = PhysicalSAGESmart(
            sage_a=float(context.run_config["sage-a"]),
            sage_b=float(context.run_config["sage-b"]),
            stale_max=int(context.run_config["smart-stale-max"]),
            partition_sizes=sizes,
            soc_min=float(context.run_config["smart-soc-min"]),
            deadline_mult=float(context.run_config["smart-deadline-mult"]),
            probe_samples=int(context.run_config["smart-probe-samples"]),
            max_waves=int(context.run_config["smart-max-waves"]),
            **smart, **common, **bcast,
        )
        print(
            f"[strategy] PhysicalSAGESmart ({strategy.variant}) "
            f"ablazione {ablation} | "
            f"(a={context.run_config['sage-a']}, "
            f"b={context.run_config['sage-b']}, "
            f"c={strategy.stale_w}) | "
            f"(E,B) decisi dal client | "
            f"soc_min={strategy.soc_min} E {strategy.epochs_min}-{strategy.epochs_cap} "
            f"deadline x{strategy.deadline_mult} probe={strategy.probe_samples} "
            + (f"rosa di {strategy.shortlist:g} k, max {strategy.max_waves} ondate | "
               if strategy.negotiation else "")
            + f"B in {strategy.batch_choices}"
        )

    elif algorithm == "sage_soc":
        strategy = PhysicalSAGEAblation(
            sage_a=float(context.run_config["sage-a"]),
            sage_b=float(context.run_config["sage-b"]),
            **common, **bcast,
        )
        print(
            f"[strategy] PhysicalSAGEAblation (a={context.run_config['sage-a']}, "
            f"b={context.run_config['sage-b']})"
        )
    elif algorithm == "fedprox":
        strategy = PhysicalFedProx(proximal_mu=proximal_mu, **common)
        print(f"[strategy] PhysicalFedProx (mu={proximal_mu})")
    elif algorithm == "fedavg":
        soc_guard = float(context.run_config["soc-guard"])
        if soc_guard > 0.0:
            strategy = PhysicalFedAvg(soc_guard=soc_guard, **common,
                                      **dict(bcast, broadcast=True))
            print(f"[strategy] PhysicalFedAvg con guardia: SoC >= {soc_guard:.0%} o in carica, "
                  f"broadcast dello stato ({bcast['broadcast_nonresp']:.0%} non risponde, "
                  f"timeout {bcast['broadcast_timeout_s']} s)")
        else:
            strategy = PhysicalFedAvg(**common)
            print("[strategy] PhysicalFedAvg")
    else:
        raise ValueError(
            f"algorithm='{algorithm}' non riconosciuto: "
            f"usa 'fedavg', 'fedprox', 'sage', 'sage_soc', 'sage_smart' "
            f"o 'escs-XY'"
        )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr, "lr-max": float(context.run_config["lr-max"])}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    try:
        csv_path = _write_round_csv(
            strategy, algorithm, seed, result,
            out_dir=str(context.run_config["results-dir"]),
        )
        print(f"\n[log] metriche per round in {csv_path}")
    except Exception as exc:
        print(f"\n[log] CSV per round non scritto: {exc}")

    s = world.stats()
    print(
        f"\n=== batteria {BATTERY_NAMES[world.battery]}: carica totale "
        f"{s['total_energy_wh']:.3f} Wh "
        f"(train {s['energy_train_wh']:.3f} + comm {s['energy_comm_wh']:.3f} "
        f"+ idle {s['energy_idle_wh']:.3f}) "
        f"| al carico P*dt: {s['energy_lin_wh']:.3f} Wh ===\n"
        + "SoC per tier: " + " | ".join(
            f"{t} {s[f'soc_{t}']:.3f} ({s[f'dead_{t}']} morti)"
            for t in sorted(k[4:] for k in s if k.startswith("soc_"))) + "\n"
        + f"workload medio {s['mean_workload']:.3f}\n"
        f"SoC medio {s['mean_soc']:.3f} | mediano {s['median_soc']:.3f} | "
        f"min {s['min_soc']:.3f} | max {s['max_soc']:.3f} | "
        f"p25 {s['p25_soc']:.3f} | p75 {s['p75_soc']:.3f}\n"
        f"SoC=0: {s['n_soc_zero']}/{s['n_clients']} | ricariche "
        f"{s['total_recharges']} | in ricarica {s['n_recharging']} | "
        f"falliti {s['n_failed']}"
    )

    if context.run_config["save-model"]:
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def _write_round_csv(strategy, algorithm: str, seed: int, history,
                     out_dir: str = "results") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{algorithm}_seed{seed}_{stamp}_{os.getpid()}.csv"

    acc_by_round: dict[int, tuple[float, float]] = dict(_ACC_BY_ROUND)

    if not acc_by_round:
        for attr in ("evaluate_metrics_serverapp",
                     "evaluate_metrics_clientapp",
                     "train_metrics_clientapp"):
            m = getattr(history, attr, None)
            if not isinstance(m, dict) or not m:
                continue
            for rnd, rec in m.items():
                try:
                    keys = list(rec.keys())
                except AttributeError:
                    continue
                acc_k = next((k for k in keys if "acc" in str(k).lower()), None)
                loss_k = next((k for k in keys if "loss" in str(k).lower()), None)
                if acc_k is None:
                    continue
                try:
                    acc_by_round[int(rnd)] = (
                        float(rec[acc_k]),
                        float(rec[loss_k]) if loss_k else 0.0,
                    )
                except (TypeError, ValueError):
                    continue
            if acc_by_round:
                break

    stats = getattr(strategy, "_round_stats", {}) or {}
    rounds = sorted(stats) if stats else sorted(acc_by_round)

    cols = ["round", "accuracy", "loss", "total_energy_wh", "energy_train_wh",
            "energy_comm_wh", "energy_lin_wh", "mean_soc", "median_soc", "min_soc", "max_soc",
            "n_soc_zero", "n_failed", "n_recharging", "total_recharges"]
    known = set(cols)
    extra = sorted({k for st in stats.values() for k in st} - known)
    cols += extra
    try:
        dev_soc = getattr(strategy, "_dev_soc", {}) or {}
        world = getattr(strategy, "world", None)
        if dev_soc:
            n_dev = len(next(iter(dev_soc.values())))
            with open(path.with_suffix(".devsoc"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["round"] + [f"soc_{c}" for c in range(n_dev)])
                for r in sorted(dev_soc):
                    w.writerow([r] + [f"{x:.5f}" for x in dev_soc[r]])
        if world is not None and hasattr(world, "device_rows"):
            part = getattr(strategy, "_participations", {}) or {}
            rows = world.device_rows()
            with open(path.with_suffix(".devinfo"), "w", newline="") as fh:
                w = csv.writer(fh)
                keys = list(rows[0]) + ["participations"] if rows else []
                w.writerow(keys)
                for row in rows:
                    w.writerow([row[k] if not isinstance(row[k], float) else f"{row[k]:.6f}"
                                for k in keys[:-1]] + [part.get(row["cid"], 0)])
    except Exception as exc:
        print(f"[log] file per device non scritti: {exc}")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rounds:
            st = stats.get(r, {})
            acc, loss = acc_by_round.get(r, (None, None))
            w.writerow([r,
                        "" if acc is None else f"{acc:.6f}",
                        "" if loss is None else f"{loss:.6f}"]
                       + [st.get(c, "") for c in cols[3:]])
    return path


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = build_model(_MODEL_NAME, num_classes(_DATASET))
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset(_DATASET)

    test_loss, test_acc = test(model, test_dataloader, device)

    _ACC_BY_ROUND[int(server_round)] = (float(test_acc), float(test_loss))

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
