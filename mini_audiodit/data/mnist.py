from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_mnist_loaders(
    root: str | Path,
    batch_size: int,
    num_workers: int = 0,
    image_size: int = 28,
) -> tuple[DataLoader, DataLoader]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tfm_list = [transforms.ToTensor()]
    if image_size != 28:
        tfm_list.insert(0, transforms.Resize(image_size))
    tfm = transforms.Compose(tfm_list)

    train = datasets.MNIST(str(root), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(root), train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader
