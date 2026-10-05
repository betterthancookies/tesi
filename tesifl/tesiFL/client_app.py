"""tesiFL: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

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
    # [B] distribuzione delle classi locali, per la divergenza JS di SAGE.
    # Su TUTTI i campioni: nel codice originale il bincount finale usava la
    # sola variabile del ciclo, cioe' l'ultimo batch.
    class_counts = [0] * 10
    for _b in trainloader:
        for _lbl in _b["label"].tolist():
            if 0 <= int(_lbl) < 10:
                class_counts[int(_lbl)] += 1

    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "class-dist": class_counts,
        # ingredienti della statistical utility di Oort; ignorati dalle altre
        # strategie, che leggono solo train_loss e num-examples
        "loss-sq-sum": loss_sq_sum,
        "loss-n": loss_n,
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
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