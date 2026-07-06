"""
Visualize Dataset Script
-------------------------
Hiển thị mẫu ảnh từ dataset cây cà chua theo dạng grid.

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
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import TomatoSplitDataset


def main():
    parser = argparse.ArgumentParser(description="Visualize mẫu ảnh từ dataset cà chua")
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
        "--output",
        type=str,
        default="results/figures/dataset_samples.png",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    image_size = config.get("image_size", 64)
    split_csv = f"data/splits/{args.split}.csv"

    # Đọc class_names nếu có
    class_names_path = "data/splits/class_names.txt"
    class_names = {}
    if os.path.isfile(class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    class_names[int(parts[0])] = parts[1]

    dataset = TomatoSplitDataset(
        split_csv=split_csv,
        image_size=image_size,
        augment=False,
        split=args.split,
    )

    n = min(args.num_images, len(dataset))
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    for i in range(n):
        img_tensor, label = dataset[i]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        label_name = class_names.get(label, str(label))
        axes_flat[i].imshow(img_np.clip(0, 1))
        axes_flat[i].set_title(label_name, fontsize=9)
        axes_flat[i].axis("off")

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle(
        f"Dataset Cây Cà Chua — Split: {args.split} ({n} ảnh, {image_size}×{image_size})",
        fontsize=13,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✅ Visualized {n} ảnh từ split '{args.split}' -> {args.output}")


if __name__ == "__main__":
    main()
