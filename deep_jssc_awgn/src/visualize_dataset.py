"""
Visualize Dataset Script
-------------------------
Hiển thị mẫu ảnh từ dataset CIFAR-10 theo dạng grid.

Sử dụng:
    python src/visualize_dataset.py --split train --num_images 16
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CIFAR10SplitDataset


def main():
    parser = argparse.ArgumentParser(description="Visualize mẫu ảnh từ dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split cần visualize",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=16,
        help="Số ảnh cần hiển thị",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cbr_1_6.yaml",
        help="Config file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/figures/dataset_samples.png",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    image_size = config.get("image_size", 64)
    split_csv = f"data/splits/{args.split}.csv"

    dataset = CIFAR10SplitDataset(
        split_csv=split_csv,
        data_root=args.data_root,
        split=args.split,
        image_size=image_size,
        augment=False,
    )

    n = min(args.num_images, len(dataset))
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    axes = axes.flatten() if nrows > 1 else [axes] * ncols

    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    for i in range(n):
        img_tensor, label = dataset[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np.clip(0, 1))
        axes[i].set_title(CIFAR10_CLASSES[label], fontsize=9)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Dataset CIFAR-10 — Split: {args.split} ({n} ảnh, {image_size}×{image_size})",
        fontsize=13,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ Visualized {n} ảnh từ split '{args.split}' -> {args.output}")


if __name__ == "__main__":
    main()
