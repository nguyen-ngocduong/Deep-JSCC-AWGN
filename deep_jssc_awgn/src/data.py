"""
Data Loading Module
-------------------
PyTorch Dataset và DataLoader cho CIFAR-10 trong project Deep JSCC.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from typing import Optional, Tuple


class CIFAR10SplitDataset(Dataset):
    """
    Dataset CIFAR-10 dựa trên file CSV split.

    Args:
        split_csv: Đường dẫn CSV chứa cột 'index', 'split'.
        data_root: Thư mục raw CIFAR-10.
        split: 'train', 'val', hoặc 'test'.
        image_size: Kích thước ảnh mục tiêu (resize).
        augment: Có dùng data augmentation hay không (chỉ cho train).
    """

    def __init__(
        self,
        split_csv: str,
        data_root: str = "data/raw",
        split: str = "train",
        image_size: int = 64,
        augment: bool = False,
    ):
        self.split = split
        self.image_size = image_size

        # Đọc CSV để lấy danh sách index
        df = pd.read_csv(split_csv)
        self.indices = df["index"].tolist()

        # Xác định transform
        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, padding=4),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        # Tải CIFAR-10 dataset gốc (không transform)
        is_train_split = split in ("train", "val")
        self.cifar = datasets.CIFAR10(
            root=data_root,
            train=is_train_split,
            download=False,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cifar_idx = self.indices[idx]
        img, label = self.cifar[cifar_idx]
        img_tensor = self.transform(img)
        return img_tensor, label


def get_dataloaders(
    config: dict,
    data_root: str = "data/raw",
    split_dir: str = "data/splits",
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train, val và test.

    Args:
        config: Config dict.
        data_root: Thư mục raw CIFAR-10.
        split_dir: Thư mục chứa CSV splits.
        augment: Có dùng augmentation cho train hay không.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    image_size = config.get("image_size", 64)
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 4)

    train_dataset = CIFAR10SplitDataset(
        split_csv=os.path.join(split_dir, "train.csv"),
        data_root=data_root,
        split="train",
        image_size=image_size,
        augment=augment,
    )
    val_dataset = CIFAR10SplitDataset(
        split_csv=os.path.join(split_dir, "val.csv"),
        data_root=data_root,
        split="val",
        image_size=image_size,
        augment=False,
    )
    test_dataset = CIFAR10SplitDataset(
        split_csv=os.path.join(split_dir, "test.csv"),
        data_root=data_root,
        split="test",
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_test_loader_only(
    config: dict,
    data_root: str = "data/raw",
    split_dir: str = "data/splits",
) -> DataLoader:
    """Tạo DataLoader chỉ cho test set."""
    image_size = config.get("image_size", 64)
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 4)

    test_dataset = CIFAR10SplitDataset(
        split_csv=os.path.join(split_dir, "test.csv"),
        data_root=data_root,
        split="test",
        image_size=image_size,
        augment=False,
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
