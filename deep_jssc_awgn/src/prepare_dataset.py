"""
Dataset Preparation Module
--------------------------
Chuẩn bị và chia dataset ảnh cây cà chua cho project Deep JSCC.

Cấu trúc thư mục data/raw mong đợi:
    data/raw/
        <class_name>/       # ví dụ: healthy, early_blight, late_blight, ...
            img001.jpg
            img002.png
            ...
    HOẶC flat (tất cả ảnh thẳng trong data/raw/, không có subdirectory):
    data/raw/
        img001.jpg
        img002.png
        ...

Đầu ra: data/splits/train.csv, val.csv, test.csv
"""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple

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

# Định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def load_config(config_path: str) -> dict:
    """Đọc file YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scan_tomato_images(data_root: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Quét toàn bộ ảnh cây cà chua trong data_root.

    Hỗ trợ 2 chế độ:
    1. Có subdirectory (class-based): mỗi thư mục con là một nhãn lớp.
    2. Flat: tất cả ảnh nằm thẳng trong data_root, nhãn = 0.

    Returns:
        paths      : danh sách đường dẫn tuyệt đối tới ảnh.
        labels     : danh sách nhãn số nguyên tương ứng.
        class_names: danh sách tên class theo thứ tự nhãn.
    """
    data_root = Path(data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Thư mục data_root không tồn tại: {data_root}\n"
            "Hãy đặt ảnh cây cà chua vào data/raw/ rồi chạy lại."
        )

    subdirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    paths: List[str] = []
    labels: List[int] = []
    class_names: List[str] = []

    if subdirs:
        # Chế độ class-based
        class_names = [d.name for d in subdirs]
        logger.info(f"Phát hiện {len(class_names)} lớp: {class_names}")
        for label_idx, subdir in enumerate(subdirs):
            imgs = [
                str(p) for p in sorted(subdir.iterdir())
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            logger.info(f"  [{label_idx}] {subdir.name}: {len(imgs)} ảnh")
            paths.extend(imgs)
            labels.extend([label_idx] * len(imgs))
    else:
        # Chế độ flat
        class_names = ["tomato"]
        flat_imgs = [
            str(p) for p in sorted(data_root.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        logger.info(f"Chế độ flat: {len(flat_imgs)} ảnh trong {data_root}")
        paths.extend(flat_imgs)
        labels.extend([0] * len(flat_imgs))

    if not paths:
        raise RuntimeError(
            f"Không tìm thấy ảnh nào trong {data_root}.\n"
            f"Các định dạng hỗ trợ: {IMAGE_EXTENSIONS}"
        )

    return paths, labels, class_names


def prepare_tomato(
    data_root: str,
    split_dir: str,
    image_size: int = 64,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Quét ảnh cây cà chua từ data_root và tạo file CSV split.

    Args:
        data_root : Thư mục chứa ảnh cây cà chua (data/raw).
        split_dir : Thư mục lưu CSV splits (data/splits).
        image_size: Kích thước ảnh mục tiêu (ghi vào summary).
        val_ratio : Tỷ lệ validation từ phần train+val.
        test_ratio: Tỷ lệ test từ toàn bộ dataset.
        seed      : Random seed để tái tạo lại được.
    """
    logger.info("Đang quét ảnh cây cà chua từ data/raw...")
    os.makedirs(split_dir, exist_ok=True)

    paths, labels, class_names = scan_tomato_images(data_root)
    total = len(paths)
    logger.info(f"Tổng số ảnh: {total}")
    logger.info(f"Danh sách lớp: {class_names}")

    # Shuffle toàn bộ dataset
    random.seed(seed)
    indices = list(range(total))
    random.shuffle(indices)

    # Chia test trước
    n_test = max(1, int(total * test_ratio))
    n_val = max(1, int(total * val_ratio))
    n_train = total - n_test - n_val

    test_indices = indices[:n_test]
    val_indices = indices[n_test: n_test + n_val]
    train_indices = indices[n_test + n_val:]

    logger.info(
        f"Phân chia: Train={len(train_indices)}, "
        f"Val={len(val_indices)}, Test={len(test_indices)}"
    )

    def write_csv(filepath: str, idx_list: List[int], split_name: str):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "label", "class_name", "split"])
            for i in idx_list:
                writer.writerow([
                    paths[i],
                    labels[i],
                    class_names[labels[i]],
                    split_name,
                ])
        logger.info(f"Đã ghi {len(idx_list)} ảnh vào {filepath}")

    write_csv(os.path.join(split_dir, "train.csv"), train_indices, "train")
    write_csv(os.path.join(split_dir, "val.csv"),   val_indices,   "val")
    write_csv(os.path.join(split_dir, "test.csv"),  test_indices,  "test")

    # Tạo dataset_summary.csv
    summary_path = os.path.join(
        os.path.dirname(split_dir), "tables", "dataset_summary.csv"
    )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary = pd.DataFrame([
        {"split": "train", "n_images": len(train_indices), "source": "tomato"},
        {"split": "val",   "n_images": len(val_indices),   "source": "tomato"},
        {"split": "test",  "n_images": len(test_indices),  "source": "tomato"},
    ])
    summary.to_csv(summary_path, index=False)
    logger.info(f"Dataset summary lưu tại {summary_path}")
    logger.info(f"Image size khi train/eval: {image_size}x{image_size}")

    # Ghi class_names.txt để tham chiếu
    class_names_path = os.path.join(split_dir, "class_names.txt")
    with open(class_names_path, "w", encoding="utf-8") as f:
        for i, name in enumerate(class_names):
            f.write(f"{i}\t{name}\n")
    logger.info(f"Class names lưu tại {class_names_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset cây cà chua cho Deep JSCC"
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
        help="Thư mục chứa ảnh cây cà chua",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Tỷ lệ validation từ toàn bộ dataset",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Tỷ lệ test từ toàn bộ dataset",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    image_size = config.get("image_size", 64)
    split_dir = "data/splits"

    logger.info(f"Config: {args.config}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Seed: {seed}, Image size: {image_size}")

    prepare_tomato(
        data_root=args.data_root,
        split_dir=split_dir,
        image_size=image_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=seed,
    )

    logger.info("✅ Chuẩn bị dataset cây cà chua xong!")


if __name__ == "__main__":
    main()
