"""tesiFL: A Flower / PyTorch app."""

import csv
import os
import time
from pathlib import Path

# [B] accuracy per round, popolata da global_evaluate: sorgente diretta, non
# dipende dai nomi dei campi di Result (che cambiano fra versioni di Flower)
_ACC_BY_ROUND: dict[int, tuple[float, float]] = {}

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from device.world_state import build_world_state
from strategies.fedavg import PhysicalFedAvg
from strategies.fedprox import PhysicalFedProx
from strategies.sage import PhysicalSAGE
from strategies.sage_ablation import PhysicalSAGEAblation
from strategies.sage_smart import PhysicalSAGESmart
from strategies.sage_smart2 import PhysicalSAGESmart2
from strategies.escs import PhysicalESCS
from tesiFL.data.partition import load_centralized_dataset, partition_sizes
from tesiFL.task import test
from tesiFL.data.model import build_model

# Create ServerApp
app = ServerApp()

# [B] nome del modello, impostato da main() e letto da global_evaluate().
# Quest'ultima e' passata come callback a strategy.start() e non riceve il
# Context, quindi non puo' leggere run_config da se'.
_MODEL_NAME = "resnet20"


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    epochs: int = context.run_config["local-epochs"]
    batch_size: int = context.run_config["batch-size"]
    seed: int = context.run_config["seed"]
    algorithm: str = str(context.run_config["algorithm"]).lower()
    proximal_mu: float = context.run_config["proximal-mu"]
    global _MODEL_NAME
    _MODEL_NAME = str(context.run_config["model-name"])

    # [B] senza questo l'init del modello e' casuale a ogni run e i confronti
    # tra algoritmi non sono validi (accuracy iniziale diversa a round 0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # [B] senza questo cuDNN sceglie kernel non deterministici e due run con
    # la stessa selezione divergono nell'accuracy (non nell'energia)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load global model
    # [B] costruito DOPO manual_seed: l'inizializzazione dei pesi consuma la
    # RNG di torch, quindi farlo prima renderebbe il seed inefficace.
    global_model = build_model(_MODEL_NAME, 10)
    arrays = ArrayRecord(global_model.state_dict())

    # Mondo fisico: stesso seed -> stessa popolazione (confronto equo tra
    # metodi, i node_id di Ray cambiano ma i cid/profili no)
    n_clients = len(grid.get_node_ids())
    capacity_mah = float(context.run_config["battery-capacity-mah"])
    k_per_round = int(context.run_config["clients-per-round"])
    tiers_on = bool(context.run_config["tiers-enabled"])
    workload_on = bool(context.run_config["workload-enabled"])
    idle_on = bool(context.run_config["idle-enabled"])
    recharge_on = bool(context.run_config["recharge-enabled"])
    recharge_avail = bool(context.run_config["recharge-available"])
    world = build_world_state(n_clients=n_clients, seed=seed,
                              capacity_mah=capacity_mah,
                              tiers_enabled=tiers_on,
                              workload_enabled=workload_on,
                              idle_enabled=idle_on,
                              recharge_enabled=recharge_on,
                              recharge_available=recharge_avail,
                              battery_variant="efficiency")
    from collections import Counter
    tiers = Counter(world.tier(c) for c in range(n_clients))
    print(f"[world] {n_clients} SuperNode | k={k_per_round} "
          f"({k_per_round / max(n_clients, 1):.0%}) | "
          f"tier {'ON ' + str(dict(tiers)) if tiers_on else f'OFF ({capacity_mah:.0f} mAh)'}")
    print(f"[world] workload {'ON ' + str(world.workload.counts_by_profile()) if workload_on else 'OFF'}"
          f" | idle {'ON' if idle_on else 'OFF'}"
          f" | ricarica {('ON, modalita ' + ('a: disponibile' if recharge_avail else 'b: unavailable')) if recharge_on else 'OFF'}")

    # Initialize strategy: FedAvg + bookkeeping fisico (Sez. 4 del piano)
    common = dict(
        world=world,
        epochs=epochs,
        batch_size=batch_size,
        k_per_round=k_per_round,
        selection_seed=seed,
        fraction_evaluate=fraction_evaluate,
    )
    if algorithm.startswith("escs"):
        # escs, escs-sd, escs-sp, escs-md, escs-mp
        suffix = algorithm.split("-")[-1] if "-" in algorithm else "sd"
        if len(suffix) != 2 or suffix[0] not in "sm" or suffix[1] not in "dp":
            raise ValueError(
                f"variante ESCS non valida: '{algorithm}' "
                "(usa escs-sd, escs-sp, escs-md, escs-mp)"
            )
        # [B] ESCS ha bisogno della latenza di training di OGNI client prima
        # del primo round (nel paper e' un dato di profilo): la si predice
        # sulla vera taglia della partizione, ricostruita qui con lo stesso
        # partitioner (stesso beta e seed) che usano i client.
        sizes = partition_sizes(
            num_partitions=n_clients,
            beta=float(context.run_config["beta"]),
            seed=seed,
        )
        strategy = PhysicalESCS(
            utility_mode=suffix[0], selection_mode=suffix[1],
            battery_mode=str(context.run_config["escs-battery"]),
            partition_sizes=sizes,
            first_round_all=bool(context.run_config["escs-first-round-all"]),
            cap_probabilistic=bool(context.run_config["escs-cap-probabilistic"]),
            min_battery=float(context.run_config["escs-min-battery"]),
            min_network_quality=float(context.run_config["escs-min-nq"]),
            **common,
        )
        print(f"[strategy] PhysicalESCS ({strategy.variant})")
    elif algorithm == "sage":
        strategy = PhysicalSAGE(
            sage_a=float(context.run_config["sage-a"]),
            sage_b=float(context.run_config["sage-b"]),
            sage_c=float(context.run_config["sage-c"]),
            **common,
        )
        print(
            f"[strategy] PhysicalSAGE (a={context.run_config['sage-a']}, "
            f"b={context.run_config['sage-b']}, c={context.run_config['sage-c']})"
        )
    elif algorithm == "sage_smart":
        strategy = PhysicalSAGESmart(
            sage_a=float(context.run_config["sage-a"]),
            sage_b=float(context.run_config["sage-b"]),
            stale_weight=float(context.run_config["smart-stale-weight"]),
            stale_max=int(context.run_config["smart-stale-max"]),
            **common,
        )

    elif algorithm == "sage_smart2":
        # [B] la taglia dei dataset e' un dato di ISCRIZIONE (Fase 0 della
        # proposta): serve al server per la deadline e per l'envelope prima
        # che il client abbia mai risposto. Stesso partitioner di ESCS.
        sizes = partition_sizes(
            num_partitions=n_clients,
            beta=float(context.run_config["beta"]),
            seed=seed,
        )
        strategy = PhysicalSAGESmart2(
            sage_a=float(context.run_config["sage-a"]),
            sage_b=float(context.run_config["sage-b"]),
            stale_weight=float(context.run_config["smart-stale-weight"]),
            stale_max=int(context.run_config["smart-stale-max"]),
            partition_sizes=sizes,
            soc_min=float(context.run_config["smart2-soc-min"]),
            epochs_min=int(context.run_config["smart2-epochs-min"]),
            deadline_mult=float(context.run_config["smart2-deadline-mult"]),
            probe_samples=int(context.run_config["smart2-probe-samples"]),
            heartbeat=bool(context.run_config["smart2-heartbeat"]),
            **common,
        )
        print(
            f"[strategy] PhysicalSAGESmart2 (a={context.run_config['sage-a']}, "
            f"b={context.run_config['sage-b']}, "
            f"c={context.run_config['smart-stale-weight']}) | "
            f"(E,B) decisi dal client | soc_min={strategy.soc_min} "
            f"E_lo={strategy.epochs_min} deadline x{strategy.deadline_mult} "
            f"probe={strategy.probe_samples} "
            f"heartbeat={'ON' if strategy.heartbeat else 'OFF'}"
        )

    elif algorithm == "sage_soc":
        # variante della tesi: SoC Peukert al posto dell'energia lineare,
        # niente termine rinnovabile (a + b = 1), niente soglia di
        # eleggibilita'. NB: non accetta sage-c.
        strategy = PhysicalSAGEAblation(
            sage_a=float(context.run_config["sage-a"]),
            sage_b=float(context.run_config["sage-b"]),
            **common,
        )
        print(
            f"[strategy] PhysicalSAGEAblation (a={context.run_config['sage-a']}, "
            f"b={context.run_config['sage-b']})"
        )
    elif algorithm == "fedprox":
        strategy = PhysicalFedProx(proximal_mu=proximal_mu, **common)
        print(f"[strategy] PhysicalFedProx (mu={proximal_mu})")
    elif algorithm == "fedavg":
        strategy = PhysicalFedAvg(**common)
        print("[strategy] PhysicalFedAvg")
    else:
        raise ValueError(
            f"algorithm='{algorithm}' non riconosciuto: "
            f"usa 'fedavg', 'fedprox', 'sage', 'sage_soc', 'sage_smart', "
            f"'sage_smart2' o 'escs-XY'"
        )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    try:
        csv_path = _write_round_csv(
            strategy, algorithm, seed, result,
            out_dir=str(context.run_config["results-dir"]),
        )
        print(f"\n[log] metriche per round in {csv_path}")
    except Exception as exc:  # il CSV non deve far fallire la simulazione
        print(f"\n[log] CSV per round non scritto: {exc}")

    s = world.stats()
    print(
        f"\n=== energia fisica totale: {s['total_energy_wh']:.3f} Wh "
        f"(train {s['energy_train_wh']:.3f} + comm {s['energy_comm_wh']:.3f} "
        f"+ idle {s['energy_idle_wh']:.3f}) "
        f"| lineare P*dt: {s['energy_lin_wh']:.3f} Wh ===\n"
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
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def _write_round_csv(strategy, algorithm: str, seed: int, history,
                     out_dir: str = "results") -> Path:
    """Una riga per round: accuracy + stato fisico cumulato.

    Le metriche derivate (round-to-accuracy alle varie soglie, energia spesa
    per arrivarci, curva energia-accuratezza) si calcolano da qui in fase di
    analisi, senza dover rieseguire le simulazioni.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{algorithm}_seed{seed}_{stamp}_{os.getpid()}.csv"

    # sorgente primaria: quanto registrato da global_evaluate
    acc_by_round: dict[int, tuple[float, float]] = dict(_ACC_BY_ROUND)

    # fallback: i dizionari per round di Result (evaluate_metrics_serverapp e
    # simili), usati solo se l'accumulatore fosse vuoto
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
    # [B] solo i round in cui c'e' stato training: dopo l'esaurimento del pool
    # (sage_soc) global_evaluate puo' aver registrato accuracy per round senza
    # partecipanti, che nel CSV sarebbero righe con energia vuota.
    rounds = sorted(stats) if stats else sorted(acc_by_round)

    cols = ["round", "accuracy", "loss", "total_energy_wh", "energy_train_wh",
            "energy_comm_wh", "energy_lin_wh", "mean_soc", "median_soc", "min_soc", "max_soc",
            "n_soc_zero", "n_failed", "n_recharging", "total_recharges"]
    # [B] colonne nuove IN CODA: i lettori esistenti accedono per nome, quindi
    # analyze_rounds.py e plot_curves.py continuano a funzionare senza modifiche.
    known = set(cols)
    extra = sorted({k for st in stats.values() for k in st} - known)
    cols += extra
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
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = build_model(_MODEL_NAME, 10)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    _ACC_BY_ROUND[int(server_round)] = (float(test_acc), float(test_loss))

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})