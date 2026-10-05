"""Dirichlet partitioning of CIFAR-10 for tesiFL.

Framework-agnostic data loading: no Flower imports here, only
flwr-datasets (partitioning utility) and torch/torchvision (tensors).
"""
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, beta: float = 0.5, seed: int = 42):
    """Load partition CIFAR10 data with Dirichlet non-IID partitioning."""
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=beta,  # 0.1 / 0.5 / 1.0
            seed=seed,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def partition_sizes(num_partitions: int, beta: float, seed: int) -> dict[int, int]:
    """Numero di campioni di TRAINING di ogni partizione, senza caricare le
    immagini in memoria piu' del necessario.

    Riproduce esattamente lo split di load_data (stesso partitioner, stesso
    train_test_split con test_size=0.2 e stesso seed), quindi il valore
    coincide con `num-examples` che il client riporta a fine training.

    Serve alle strategie che, come ESCS, hanno bisogno della latenza di
    training di OGNI client prima del primo round (nel paper e' un dato di
    profilo, `total_train_latency`).
    """
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions, partition_by="label", alpha=beta, seed=seed,
    )
    fds_local = FederatedDataset(
        dataset="uoft-cs/cifar10", partitioners={"train": partitioner},
    )
    sizes: dict[int, int] = {}
    for pid in range(num_partitions):
        part = fds_local.load_partition(pid)
        sizes[pid] = len(part.train_test_split(test_size=0.2, seed=seed)["train"])
    return sizes


def load_centralized_dataset():
    """Load test set and return dataloader."""
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)