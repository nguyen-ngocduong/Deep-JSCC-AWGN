"""
Data Loading Module
-------------------
PyTorch Dataset và DataLoader cho tập dữ liệu ảnh cây cà chua trong project Deep JSCC.

Cấu trúc thư mục data/raw mong đợi:
    data/raw/
        <class_name_1>/
            img1.jpg
            img2.png
            ...
        <class_name_2>/
            ...
    hoặc flat (tất cả ảnh nằm trực tiếp trong data/raw/, không có subdirectory).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _scan_images(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Quét toàn bộ ảnh trong data_root.

    Hỗ trợ 2 chế độ:
    1. Có subdirectory (class-based): mỗi thư mục con là một nhãn lớp.
    2. Flat: tất cả ảnh nằm thẳng trong data_root, nhãn = 0.

    Returns:
        paths: danh sách đường dẫn tuyệt đối tới ảnh.
        labels: danh sách nhãn số nguyên tương ứng.
        class_names: danh sách tên class theo thứ tự nhãn.
    """
    data_root = Path(data_root)
    subdirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    paths: List[str] = []
    labels: List[int] = []
    class_names: List[str] = []

    if subdirs:
        # Chế độ class-based (ImageFolder style)
        class_names = [d.name for d in subdirs]
        for label_idx, subdir in enumerate(subdirs):
            for img_path in sorted(subdir.iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(img_path))
                    labels.append(label_idx)
    else:
        # Chế độ flat — tất cả ảnh cùng nhãn "tomato" (label=0)
        class_names = ["tomato"]
        for img_path in sorted(data_root.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(img_path))
                labels.append(0)

    return paths, labels, class_names


class TomatoSplitDataset(Dataset):
    """
    Dataset ảnh cây cà chua dựa trên file CSV split.

    Args:
        split_csv: Đường dẫn CSV chứa cột 'filepath', 'label', 'split'.
        image_size: Kích thước ảnh mục tiêu (resize).
        augment: Có dùng data augmentation hay không (chỉ cho train).
    """

    def __init__(
        self,
        split_csv: str,
        image_size: int = 64,
        augment: bool = False,
        split: str = "train",
    ):
        self.split = split
        self.image_size = image_size

        # Đọc CSV
        df = pd.read_csv(split_csv)
        self.filepaths: List[str] = df["filepath"].tolist()
        self.labels: List[int] = df["label"].tolist()

        # Transform
        if augment and split == "train":
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.filepaths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, self.labels[idx]


def get_dataloaders(
    config: dict,
    data_root: str = "data/raw",
    split_dir: str = "data/splits",
    augment: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train, val và test từ tập cà chua.

    Args:
        config: Config dict.
        data_root: Thư mục raw (chỉ dùng để tham chiếu, paths đã lưu trong CSV).
        split_dir: Thư mục chứa CSV splits.
        augment: Có dùng augmentation cho train hay không.

    Returns:
        Tuple (train_loader, val_loader, test_loader).
    """
    image_size = config.get("image_size", 64)
    batch_size = config.get("batch_size", 64)
    num_workers = config.get("num_workers", 4)

    train_dataset = TomatoSplitDataset(
        split_csv=os.path.join(split_dir, "train.csv"),
        image_size=image_size,
        augment=augment,
        split="train",
    )
    val_dataset = TomatoSplitDataset(
        split_csv=os.path.join(split_dir, "val.csv"),
        image_size=image_size,
        augment=False,
        split="val",
    )
    test_dataset = TomatoSplitDataset(
        split_csv=os.path.join(split_dir, "test.csv"),
        image_size=image_size,
        augment=False,
        split="test",
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

    test_dataset = TomatoSplitDataset(
        split_csv=os.path.join(split_dir, "test.csv"),
        image_size=image_size,
        augment=False,
        split="test",
    )
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
