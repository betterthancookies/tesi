"""tesiFL: local training/evaluation functions (framework-agnostic)."""
import torch


def train(net, trainloader, epochs, lr, device, proximal_mu: float = 0.0,
          max_steps: int | None = None, stats: dict | None = None):
    """Train the model on the training set.

    proximal_mu > 0 attiva il termine prossimale di FedProx:
        loss = CE + (mu/2) * ||w - w_global||^2
    dove w_global sono i pesi ricevuti dal server a inizio round (congelati,
    detached: il gradiente non ci passa attraverso). mu=0 -> FedAvg puro,
    nessun costo computazionale aggiuntivo.

    max_steps  interrompe il training dopo quel numero di passi di SGD, anche
               a meta' epoca (Fase 6 della proposta: early stopping quando il
               budget energetico finisce). None = epoche complete, come prima.
    stats      se e' un dict, ci scrive "steps" (passi eseguiti) e
               "last_loss" (loss media dell'ultima epoca, anche parziale):
               servono a sage_smart. Il valore di ritorno non cambia.
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
    steps = 0
    last_sum, last_n = 0.0, 0
    for _ep in range(epochs):
        # [B] controllo PRIMA dell'azzeramento: se il budget finisce
        # esattamente a fine epoca, le statistiche restano quelle dell'epoca
        # appena chiusa invece di un'epoca vuota
        if max_steps is not None and steps >= max_steps:
            break
        # statistiche riferite all'ULTIMA epoca: riflettono lo stato del
        # modello a fine round, non la media del percorso
        sq_sum, n_seen = 0.0, 0
        last_sum, last_n = 0.0, 0
        for batch in trainloader:
            if max_steps is not None and steps >= max_steps:
                break
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            per_sample = criterion_ns(net(images), labels)
            with torch.no_grad():
                sq_sum += float(torch.sum(per_sample ** 2).item())
                n_seen += int(per_sample.numel())
                last_sum += float(torch.sum(per_sample).item())
                last_n += int(per_sample.numel())
            loss = per_sample.mean()
            if use_prox:
                prox_term = 0.0
                for p, g in zip(net.parameters(), global_params):
                    prox_term = prox_term + torch.sum((p - g) ** 2)
                loss = loss + (proximal_mu / 2.0) * prox_term
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
    # [B] diviso per i passi EFFETTIVI: con epoche complete steps vale
    # epochs * len(trainloader), cioe' esattamente il denominatore di prima
    avg_trainloss = running_loss / max(1, steps)
    if stats is not None:
        stats["steps"] = steps
        stats["last_loss"] = last_sum / last_n if last_n else float("nan")
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
