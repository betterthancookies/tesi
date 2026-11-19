from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, ToTensor)

DATASETS = {
    "cifar10": ("uoft-cs/cifar10", "label", 10),
    "cifar100": ("uoft-cs/cifar100", "fine_label", 100),
}

fds = None

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_transforms = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(),
                            ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def num_classes(dataset: str = "cifar10") -> int:
    return _spec(dataset)[2]


def _spec(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(f"dataset sconosciuto: {dataset} (usa {', '.join(DATASETS)})")
    return DATASETS[dataset]


def _uniform(ds, label_col: str):
    if label_col != "label":
        ds = ds.rename_column(label_col, "label")
    return ds.select_columns(["img", "label"])


def apply_transforms(batch):
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def apply_train_transforms(batch):
    batch["img"] = [train_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int, batch_size: int, beta: float = 0.5,
              seed: int = 42, dataset: str = "cifar10", augment: bool = False):
    global fds
    hf_name, label_col, _ = _spec(dataset)
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by=label_col,
            alpha=beta,
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
    hf_name, label_col, _ = _spec(dataset)
    test_dataset = _uniform(load_dataset(hf_name, split="test"), label_col)
    dataset_t = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset_t, batch_size=128)
