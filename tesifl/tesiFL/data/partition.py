"""Dirichlet partitioning of CIFAR-10 / CIFAR-100 for tesiFL.

Framework-agnostic data loading: no Flower imports here, only
flwr-datasets (partitioning utility) and torch/torchvision (tensors).

Il dataset si sceglie con `dataset` (experiment.toml, [model].dataset):
    cifar10    uoft-cs/cifar10    etichetta "label"       10 classi
    cifar100   uoft-cs/cifar100   etichetta "fine_label"  100 classi
[B] l'etichetta viene rinominata in "label" e le altre colonne scartate
(CIFAR-100 ha anche "coarse_label"): il resto del codice legge sempre
batch["img"] e batch["label"], qualunque sia il dataset.
"""
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)

# nome -> (dataset Hugging Face, colonna dell'etichetta, numero di classi)
DATASETS = {
    "cifar10": ("uoft-cs/cifar10", "label", 10),
    "cifar100": ("uoft-cs/cifar100", "fine_label", 100),
}

fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# [B] 19/10: data augmentation standard per CIFAR (He et al. 2016): crop
# 32x32 da un'immagine con 4 pixel di padding e flip orizzontale. Solo sul
# TRAIN dei client (augment = true in [model]); test locale e test centrale
# restano senza. Motivo: senza augmentation ResNet-18 su CIFAR-100 memorizzava
# i dati di training entro il round ~100 (loss locale 0.002 con accuracy di
# test al 37%, campagna ResNet18_C100_bcast).
train_transforms = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(),
                            ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def num_classes(dataset: str = "cifar10") -> int:
    return _spec(dataset)[2]


def _spec(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(f"dataset sconosciuto: {dataset} (usa {', '.join(DATASETS)})")
    return DATASETS[dataset]


def _uniform(ds, label_col: str):
    """Colonne uniformi fra i dataset: solo "img" e "label"."""
    if label_col != "label":
        ds = ds.rename_column(label_col, "label")
    return ds.select_columns(["img", "label"])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def apply_train_transforms(batch):
    """Come apply_transforms, con crop e flip casuali (RNG globale di torch,
    quindi riproducibile col seed per (client, round) di client_app)."""
    batch["img"] = [train_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, beta: float = 0.5,
              seed: int = 42, dataset: str = "cifar10", augment: bool = False):
    """Load partition data with Dirichlet non-IID partitioning.

    augment=True: crop e flip casuali sul solo split di train.
    """
    global fds
    hf_name, label_col, _ = _spec(dataset)
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=label_col,
            alpha=beta,  # 0.1 / 0.5 / 1.0
            seed=seed,
        )
        fds = FederatedDataset(
            dataset=hf_name,
            partitioners={"train": partitioner},
        )
    partition = _uniform(fds.load_partition(partition_id), label_col)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=seed)
    train = partition_train_test["train"].with_transform(
        apply_train_transforms if augment else apply_transforms)
    test = partition_train_test["test"].with_transform(apply_transforms)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test, batch_size=batch_size)
    return trainloader, testloader


def partition_sizes(num_partitions: int, beta: float, seed: int,
                    dataset: str = "cifar10") -> dict[int, int]:
    """Numero di campioni di TRAINING di ogni partizione, senza caricare le
    immagini in memoria piu' del necessario.

    Riproduce esattamente lo split di load_data (stesso partitioner, stesso
    train_test_split con test_size=0.2 e stesso seed), quindi il valore
    coincide con `num-examples` che il client riporta a fine training.

    Serve alle strategie che, come ESCS, hanno bisogno della latenza di
    training di OGNI client prima del primo round (nel paper e' un dato di
    profilo, `total_train_latency`).
    """
    hf_name, label_col, _ = _spec(dataset)
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions, partition_by=label_col, alpha=beta, seed=seed,
    )
    fds_local = FederatedDataset(
        dataset=hf_name, partitioners={"train": partitioner},
    )
    sizes: dict[int, int] = {}
    for pid in range(num_partitions):
        part = fds_local.load_partition(pid)
        sizes[pid] = len(part.train_test_split(test_size=0.2, seed=seed)["train"])
    return sizes


def load_centralized_dataset(dataset: str = "cifar10"):
    """Load test set and return dataloader."""
    hf_name, label_col, _ = _spec(dataset)
    test_dataset = _uniform(load_dataset(hf_name, split="test"), label_col)
    dataset_t = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset_t, batch_size=128)
