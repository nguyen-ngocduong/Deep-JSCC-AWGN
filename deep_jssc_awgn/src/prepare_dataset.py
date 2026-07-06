"""
Dataset Preparation Module
--------------------------
Chuẩn bị và chia dataset ảnh cho project Deep JSCC.

Hỗ trợ:
- CIFAR-10 (tải tự động qua torchvision)
- Dataset ảnh tùy chỉnh (thư mục ảnh)

Đầu ra: data/splits/train.csv, val.csv, test.csv
"""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Đọc file YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_cifar10(
    data_root: str,
    split_dir: str,
    image_size: int = 64,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Tải CIFAR-10 và tạo file CSV split.

    CIFAR-10 gồm 50000 ảnh train và 10000 ảnh test (32x32 RGB).
    Ảnh sẽ được resize lên image_size x image_size khi load.

    Args:
        data_root: Thư mục lưu raw CIFAR-10.
        split_dir: Thư mục lưu CSV splits.
        image_size: Kích thước ảnh mục tiêu.
        val_ratio: Tỷ lệ validation từ training set.
        seed: Random seed để tái tạo lại được.
    """
    import torchvision

    logger.info("Đang tải CIFAR-10...")
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    # Tải dataset (không transform, chỉ để lấy danh sách)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True
    )

    classes = train_dataset.classes
    logger.info(f"Classes: {classes}")
    logger.info(f"Train: {len(train_dataset)} ảnh, Test: {len(test_dataset)} ảnh")

    # Tạo danh sách (index, label, split)
    random.seed(seed)
    all_train_indices = list(range(len(train_dataset)))
    random.shuffle(all_train_indices)

    n_val = int(len(all_train_indices) * val_ratio)
    val_indices = all_train_indices[:n_val]
    train_indices = all_train_indices[n_val:]

    # Ghi CSV
    def write_csv(filepath: str, indices, dataset, split_name: str):
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "label", "class_name", "split", "source"])
            for idx in indices:
                label = dataset.targets[idx]
                class_name = classes[label]
                writer.writerow([idx, label, class_name, split_name, "cifar10"])
        logger.info(f"Đã ghi {len(indices)} ảnh vào {filepath}")

    write_csv(
        os.path.join(split_dir, "train.csv"),
        train_indices, train_dataset, "train"
    )
    write_csv(
        os.path.join(split_dir, "val.csv"),
        val_indices, train_dataset, "val"
    )

    test_indices = list(range(len(test_dataset)))
    write_csv(
        os.path.join(split_dir, "test.csv"),
        test_indices, test_dataset, "test"
    )

    # Tạo dataset_summary.csv
    summary_path = os.path.join(
        os.path.dirname(split_dir), "tables", "dataset_summary.csv"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = pd.DataFrame([
        {"split": "train", "n_images": len(train_indices), "source": "CIFAR-10"},
        {"split": "val", "n_images": len(val_indices), "source": "CIFAR-10"},
        {"split": "test", "n_images": len(test_indices), "source": "CIFAR-10"},
    ])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Dataset summary lưu tại {summary_path}")
    logger.info(f"Image size khi train/eval: {image_size}x{image_size}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset CIFAR-10 cho Deep JSCC"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cbr_1_6.yaml",
        help="Đường dẫn file YAML config",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw",
        help="Thư mục lưu raw dataset",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Tỷ lệ validation từ training set",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    image_size = config.get("image_size", 64)
    split_dir = "data/splits"

    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {seed}, Image size: {image_size}")

    prepare_cifar10(
        data_root=args.data_root,
        split_dir=split_dir,
        image_size=image_size,
        val_ratio=args.val_ratio,
        seed=seed,
    )

    logger.info("✅ Chuẩn bị dataset xong!")


if __name__ == "__main__":
    main()
