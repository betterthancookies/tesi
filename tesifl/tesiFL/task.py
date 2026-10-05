"""tesiFL: local training/evaluation functions (framework-agnostic)."""
import torch


def train(net, trainloader, epochs, lr, device, proximal_mu: float = 0.0):
    """Train the model on the training set.

    proximal_mu > 0 attiva il termine prossimale di FedProx:
        loss = CE + (mu/2) * ||w - w_global||^2
    dove w_global sono i pesi ricevuti dal server a inizio round (congelati,
    detached: il gradiente non ci passa attraverso). mu=0 -> FedAvg puro,
    nessun costo computazionale aggiuntivo.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # [B] snapshot dei pesi globali PRIMA di qualunque step locale
    use_prox = proximal_mu > 0.0
    if use_prox:
        global_params = [p.detach().clone() for p in net.parameters()]

    # [B] reduction='none' per avere la loss dei singoli campioni: serve alla
    # statistical utility di Oort (|B|*sqrt(mean(Loss^2))). Il backward usa
    # comunque la media, quindi FedAvg/FedProx sono numericamente invariati.
    criterion_ns = torch.nn.CrossEntropyLoss(reduction="none").to(device)

    net.train()
    running_loss = 0.0
    sq_sum, n_seen = 0.0, 0
    for _ep in range(epochs):
        # statistiche riferite all'ULTIMA epoca: riflettono lo stato del
        # modello a fine round, non la media del percorso
        sq_sum, n_seen = 0.0, 0
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            per_sample = criterion_ns(net(images), labels)
            with torch.no_grad():
                sq_sum += float(torch.sum(per_sample ** 2).item())
                n_seen += int(per_sample.numel())
            loss = per_sample.mean()
            if use_prox:
                prox_term = 0.0
                for p, g in zip(net.parameters(), global_params):
                    prox_term = prox_term + torch.sum((p - g) ** 2)
                loss = loss + (proximal_mu / 2.0) * prox_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    # (loss media, somma dei quadrati per campione, n campioni dell'ultima epoca)
    return avg_trainloss, sq_sum, n_seen


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
