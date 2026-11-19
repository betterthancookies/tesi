import dataclasses

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from torch.utils.data import DataLoader, Subset

from device.fuel_gauge import RECORD_KEY as DEVICE_KEY
from device.fuel_gauge import DeviceReading
from tesiFL.client_policy import (
    Envelope, accepts_proposal, affordable_steps, can_afford_minimum, decide,
)
from tesiFL.data.partition import load_data, num_classes
from tesiFL.data.model import build_model
from tesiFL.task import test as test_fn
from tesiFL.task import train as train_fn

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    dataset = str(context.run_config["dataset"])
    n_cls = num_classes(dataset)
    augment = bool(context.run_config["augment"])

    model = build_model(str(context.run_config["model-name"]), n_cls)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    config = msg.content["config"]
    epochs = config["epochs"]
    batch_size = config["batch-size"]
    proximal_mu = float(config["proximal-mu"]) if "proximal-mu" in config else 0.0

    partition_id = (
        int(config["cid"]) if "cid" in config else context.node_config["partition-id"]
    )
    num_partitions = context.node_config["num-partitions"]
    beta = context.run_config["beta"]
    seed = context.run_config["seed"]
    torch.manual_seed(seed + 1000 * int(config["server-round"]) + partition_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if Envelope.present_in(config):
        return _train_client_driven(msg, model, device, config, partition_id,
                                    num_partitions, beta, seed, proximal_mu,
                                    dataset_name=dataset, augment=augment)

    trainloader, _ = load_data(partition_id, num_partitions, batch_size, beta, seed,
                               dataset=dataset, augment=augment)

    train_loss, loss_sq_sum, loss_n = train_fn(
        model,
        trainloader,
        epochs,
        config["lr"],
        device,
        proximal_mu=proximal_mu,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "class-dist": _class_counts(trainloader, n_cls),
        "loss-sq-sum": loss_sq_sum,
        "loss-n": loss_n,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


def _class_counts(loader, n_classes: int = 10) -> list[int]:
    counts = [0] * n_classes
    for _b in loader:
        for _lbl in _b["label"].tolist():
            if 0 <= int(_lbl) < n_classes:
                counts[int(_lbl)] += 1
    return counts


def _probe_loss(model, dataset, n_probe: int, device, generator) -> float:
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
                         proximal_mu: float, dataset_name: str = "cifar10",
                         augment: bool = False) -> Message:
    if DEVICE_KEY not in msg.content:
        raise KeyError(
            "sage_smart: il messaggio non contiene i sensori del device "
            f"(record '{DEVICE_KEY}'): la strategia deve allegarli"
        )
    reading = DeviceReading.from_record(msg.content[DEVICE_KEY])
    env = Envelope.from_config(config)
    server_round = int(config["server-round"])

    n_cls = num_classes(dataset_name)
    trainloader, _ = load_data(partition_id, num_partitions, env.batches[0],
                               beta, seed, dataset=dataset_name, augment=augment)
    dataset = trainloader.dataset
    n = len(dataset)
    env = dataclasses.replace(env, probe_samples=min(env.probe_samples, n))

    base = {
        "smart-soc-start": float(reading.soc),
        "smart-workload": float(reading.workload),
        "smart-charging": int(reading.charging),
    }

    def optout(probe: int, loss_pre: float, e_target: int) -> Message:
        b = env.batches[0]
        dt = reading.work_time_s(0, b, n, probe)
        _, soc_final = reading.soc_after(dt, b)
        metrics = {
            **base,
            "num-examples": 0,
            "train_loss": 0.0,
            "smart-optout": 1,
            "smart-epochs": 0,
            "smart-batch": int(b),
            "smart-steps": 0,
            "smart-probe": int(probe),
            "smart-soc-final": float(soc_final),
            "smart-e-target": int(e_target),
        }
        if loss_pre == loss_pre:
            metrics["smart-loss-pre"] = float(loss_pre)
        content = RecordDict({"metrics": MetricRecord(metrics)})
        return Message(content=content, reply_to=msg)

    if not can_afford_minimum(reading, env, n):
        return optout(probe=0, loss_pre=float("nan"), e_target=0)

    model.to(device)
    gen = torch.Generator().manual_seed(seed + 1000 * server_round + partition_id)
    loss_pre = _probe_loss(model, dataset, env.probe_samples, device, gen)

    plan = decide(reading, env, n, loss_pre, n_classes=n_cls)
    if plan.optout:
        return optout(probe=env.probe_samples, loss_pre=loss_pre,
                      e_target=plan.e_target)

    loader = DataLoader(dataset, batch_size=plan.batch, shuffle=True)
    spe = len(loader)
    planned = plan.epochs * spe
    if plan.steps_cap > 0:
        planned = min(planned, plan.steps_cap)
    budget = affordable_steps(reading, env, n, plan.batch, cap=planned)
    stats: dict = {}
    train_loss, loss_sq_sum, loss_n = train_fn(
        model, loader, plan.epochs, config["lr"], device,
        proximal_mu=proximal_mu, max_steps=budget, stats=stats,
    )
    steps = int(stats["steps"])

    dt = reading.work_time_s(steps / spe, plan.batch, n, env.probe_samples)
    _, soc_final = reading.soc_after(dt, plan.batch)

    metrics = {
        **base,
        "train_loss": train_loss,
        "num-examples": n,
        "class-dist": _class_counts(loader, n_cls),
        "loss-sq-sum": loss_sq_sum,
        "loss-n": loss_n,
        "smart-optout": 0,
        "smart-epochs": int(plan.epochs),
        "smart-batch": int(plan.batch),
        "smart-steps": steps,
        "smart-steps-budget": int(budget),
        "smart-budget-hit": int(steps < planned),
        "smart-probe": int(env.probe_samples),
        "smart-soc-final": float(soc_final),
        "smart-time-s": float(dt),
        "smart-deadline-ok": int(dt <= env.deadline_s),
        "smart-e-target": int(plan.e_target),
        "smart-utility": float(plan.utility),
        "smart-loss-pre": float(loss_pre),
        "smart-loss-post": float(stats["last_loss"]),
    }
    if env.partial_work:
        metrics["smart-partial"] = int(plan.steps_cap > 0)
        metrics["smart-weight"] = int(min(n, steps * plan.batch))
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()),
                          "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.query()
def query(msg: Message, context: Context):
    cfg = msg.content["config"]
    dev = msg.content[DEVICE_KEY]
    reading = DeviceReading.from_record(dev)
    n = int(dev["dataset-size"])
    metrics = {
        "smart-soc": float(reading.soc),
        "smart-workload": float(reading.workload),
        "smart-charging": int(reading.charging),
        "smart-accept": int(accepts_proposal(reading, Envelope.from_config(cfg), n)),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    dataset = str(context.run_config["dataset"])

    model = build_model(str(context.run_config["model-name"]), num_classes(dataset))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    _, valloader = load_data(partition_id, num_partitions, batch_size, beta, seed,
                             dataset=dataset)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
