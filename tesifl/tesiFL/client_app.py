"""tesiFL: A Flower / PyTorch app."""

import dataclasses

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader, Subset

from device.fuel_gauge import RECORD_KEY as DEVICE_KEY
from device.fuel_gauge import DeviceReading
from tesiFL.client_policy import (
    Envelope, affordable_steps, can_afford_minimum, decide,
)
from tesiFL.data.partition import load_data
from tesiFL.data.model import build_model
from tesiFL.task import test as test_fn
from tesiFL.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load the model and initialize it with the received weights
    model = build_model(str(context.run_config["model-name"]), 10)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # (E,B) del round: da PhysicalFedAvg.configure_train, NON da run_config --
    # e' esattamente il punto della tesi (decisione client-driven, veicolata
    # dal server per il baseline FedAvg fisso; con YourMethod sara' il client
    # stesso a deciderli, ma il canale resta questo ConfigRecord per round)
    config = msg.content["config"]
    epochs = config["epochs"]
    batch_size = config["batch-size"]
    # mu del round: presente solo con PhysicalFedProx, altrimenti 0.0 (=FedAvg)
    proximal_mu = float(config["proximal-mu"]) if "proximal-mu" in config else 0.0

    # Load the data
    # [B] il cid arriva dal server (PhysicalFedAvg.configure_train) ed e'
    # l'indice con cui il mondo fisico conosce questo client: usarlo per la
    # partizione garantisce che dati e profilo fisico restino accoppiati.
    # Fallback su node_config solo per strategie che non lo inviano.
    partition_id = (
        int(config["cid"]) if "cid" in config else context.node_config["partition-id"]
    )
    num_partitions = context.node_config["num-partitions"]
    beta = context.run_config["beta"]
    seed = context.run_config["seed"]
    # [B] seed per (client, round): riproducibile ma non identico tra client
    torch.manual_seed(seed + 1000 * int(config["server-round"]) + partition_id)
    # [B] il training gira in un processo Ray separato: le flag cuDNN del
    # server non lo raggiungono e vanno impostate anche qui
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # sage_smart2: il server ha mandato una PROPOSTA, non un'assegnazione
    if Envelope.present_in(config):
        return _train_client_driven(msg, model, device, config, partition_id,
                                    num_partitions, beta, seed, proximal_mu)

    trainloader, _ = load_data(partition_id, num_partitions, batch_size, beta, seed)

    # Call the training function
    train_loss, loss_sq_sum, loss_n = train_fn(
        model,
        trainloader,
        epochs,
        config["lr"],
        device,
        proximal_mu=proximal_mu,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "class-dist": _class_counts(trainloader),
        # ingredienti della statistical utility di Oort; ignorati dalle altre
        # strategie, che leggono solo train_loss e num-examples
        "loss-sq-sum": loss_sq_sum,
        "loss-n": loss_n,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _class_counts(loader, n_classes: int = 10) -> list[int]:
    """Distribuzione delle classi locali, per la divergenza JS di SAGE.

    [B] su TUTTI i campioni: nel codice originale il bincount finale usava la
    sola variabile del ciclo, cioe' l'ultimo batch.
    """
    counts = [0] * n_classes
    for _b in loader:
        for _lbl in _b["label"].tolist():
            if 0 <= int(_lbl) < n_classes:
                counts[int(_lbl)] += 1
    return counts


def _probe_loss(model, dataset, n_probe: int, device, generator) -> float:
    """Loss del modello GLOBALE su n_probe campioni locali estratti a caso.

    Fase 5, passo 2 della proposta: alta loss = i dati di questo client sono
    ancora mal rappresentati nel modello globale = vale la pena lavorare.
    [B] in eval() e senza gradiente: in train() il forward aggiornerebbe le
    statistiche di BatchNorm, e l'update inviato al server ne porterebbe
    traccia anche se il client poi non si allenasse.
    """
    idx = torch.randperm(len(dataset), generator=generator)[:n_probe].tolist()
    loader = DataLoader(Subset(dataset, idx), batch_size=128)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            tot += float(criterion(model(images), labels).item())
            n += int(labels.numel())
    model.train()
    return tot / max(1, n)


def _train_client_driven(msg: Message, model, device, config, partition_id: int,
                         num_partitions: int, beta: float, seed: int,
                         proximal_mu: float) -> Message:
    """sage_smart2: il client decide (E, B) dentro l'envelope del server.

    Fasi 5-7 della proposta:
      5. rilegge il proprio stato (late binding), stima la loss locale,
         sceglie (E, B) oppure si ritira;
      6. si allena con una guardia sul budget energetico;
      7. risponde con l'update e con cio' che ha fatto davvero: E e B scelti,
         passi eseguiti, SoC iniziale e finale, loss prima e dopo.

    Il SoC finale e' quello che il fuel gauge del device prevede per il
    lavoro svolto (training + upload): vedi device/fuel_gauge.py.
    """
    if DEVICE_KEY not in msg.content:
        raise KeyError(
            "sage_smart2: il messaggio non contiene i sensori del device "
            f"(record '{DEVICE_KEY}'): la strategia deve allegarli"
        )
    # [B] LATE BINDING: la lettura e' quella di adesso, non quella che il
    # server aveva al momento della selezione
    reading = DeviceReading.from_record(msg.content[DEVICE_KEY])
    env = Envelope.from_config(config)
    server_round = int(config["server-round"])

    trainloader, _ = load_data(partition_id, num_partitions, env.batches[0],
                               beta, seed)
    dataset = trainloader.dataset
    n = len(dataset)
    env = dataclasses.replace(env, probe_samples=min(env.probe_samples, n))

    base = {
        "s2-soc-start": float(reading.soc),
        "s2-workload": float(reading.workload),
        "s2-charging": int(reading.charging),
    }

    def optout(probe: int, loss_pre: float, e_target: int) -> Message:
        """Ritiro dal round: nessun update, la riserva resta intatta.

        [B] num-examples = 0: il server lo esclude dalla media, ma addebita
        comunque la comunicazione (il modello e' stato scaricato) e, se
        c'e' stata, la stima della loss.
        """
        b = env.batches[0]
        dt = reading.work_time_s(0, b, n, probe)
        _, soc_final = reading.soc_after(dt, b)
        metrics = {
            **base,
            "num-examples": 0,
            "train_loss": 0.0,
            "s2-optout": 1,
            "s2-epochs": 0,
            "s2-batch": int(b),
            "s2-steps": 0,
            "s2-probe": int(probe),
            "s2-soc-final": float(soc_final),
            "s2-e-target": int(e_target),
        }
        if loss_pre == loss_pre:          # non NaN
            metrics["s2-loss-pre"] = float(loss_pre)
        content = RecordDict({"metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    # 5a. il minimo e' sostenibile? Se no, inutile anche stimare la loss
    if not can_afford_minimum(reading, env, n):
        return optout(probe=0, loss_pre=float("nan"), e_target=0)

    # 5b. utilita' statistica: loss del modello globale sui dati locali
    model.to(device)
    gen = torch.Generator().manual_seed(seed + 1000 * server_round + partition_id)
    loss_pre = _probe_loss(model, dataset, env.probe_samples, device, gen)

    # 5c. decisione
    plan = decide(reading, env, n, loss_pre)
    if plan.optout:
        return optout(probe=env.probe_samples, loss_pre=loss_pre,
                      e_target=plan.e_target)

    # 6. training con guardia sul budget energetico
    loader = DataLoader(dataset, batch_size=plan.batch, shuffle=True)
    spe = len(loader)                        # ceil(n / B): passi per epoca
    planned = plan.epochs * spe
    budget = affordable_steps(reading, env, n, plan.batch, cap=planned)
    stats: dict = {}
    train_loss, loss_sq_sum, loss_n = train_fn(
        model, loader, plan.epochs, config["lr"], device,
        proximal_mu=proximal_mu, max_steps=budget, stats=stats,
    )
    steps = int(stats["steps"])

    # 7. SoC finale previsto dal fuel gauge per il lavoro EFFETTIVO
    dt = reading.work_time_s(steps / spe, plan.batch, n, env.probe_samples)
    _, soc_final = reading.soc_after(dt, plan.batch)

    metrics = {
        **base,
        "train_loss": train_loss,
        "num-examples": n,
        "class-dist": _class_counts(loader),
        "loss-sq-sum": loss_sq_sum,
        "loss-n": loss_n,
        "s2-optout": 0,
        "s2-epochs": int(plan.epochs),
        "s2-batch": int(plan.batch),
        "s2-steps": steps,
        "s2-steps-budget": int(budget),
        "s2-budget-hit": int(steps < planned),
        "s2-probe": int(env.probe_samples),
        "s2-soc-final": float(soc_final),
        "s2-time-s": float(dt),
        "s2-deadline-ok": int(dt <= env.deadline_s),
        "s2-e-target": int(plan.e_target),
        "s2-utility": float(plan.utility),
        "s2-loss-pre": float(loss_pre),
        "s2-loss-post": float(stats["last_loss"]),
    }
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()),
                          "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = build_model(str(context.run_config["model-name"]), 10)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    eval_config = msg.content["config"] if "config" in msg.content else {}
    partition_id = (
        int(eval_config["cid"])
        if "cid" in eval_config
        else context.node_config["partition-id"]
    )
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    beta = context.run_config["beta"]
    seed = context.run_config["seed"]
    torch.manual_seed(seed)
    _, valloader = load_data(partition_id, num_partitions, batch_size, beta, seed)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)