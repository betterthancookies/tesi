import torch

from tesiFL.client_policy import MOMENTUM


def train(net, trainloader, epochs, lr, device, proximal_mu: float = 0.0,
          max_steps: int | None = None, stats: dict | None = None):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=MOMENTUM)

    use_prox = proximal_mu > 0.0
    if use_prox:
        global_params = [p.detach().clone() for p in net.parameters()]

    criterion_ns = torch.nn.CrossEntropyLoss(reduction="none").to(device)

    net.train()
    running_loss = 0.0
    sq_sum, n_seen = 0.0, 0
    steps = 0
    last_sum, last_n = 0.0, 0
    for _ep in range(epochs):
        if max_steps is not None and steps >= max_steps:
            break
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
    avg_trainloss = running_loss / max(1, steps)
    if stats is not None:
        stats["steps"] = steps
        stats["last_loss"] = last_sum / last_n if last_n else float("nan")
    return avg_trainloss, sq_sum, n_seen


def test(net, testloader, device):
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
