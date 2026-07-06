"""
Plot Results Script
-------------------
Sinh toàn bộ hình vẽ và bảng biểu cho báo cáo từ các file CSV đã được tạo.

Hình sinh ra:
  - results/figures/final_psnr_vs_snr.png    (PSNR vs SNR với error bar)
  - results/figures/final_ssim_vs_snr.png    (SSIM vs SNR với error bar)
  - results/figures/jscc_cbr_comparison_psnr.png (So sánh các CBR)
  - results/figures/reconstruction_grid.png  (Ảnh khôi phục trực quan)

Bảng sinh ra:
  - results/tables/final_psnr_ssim_table.csv
  - results/tables/final_simulation_settings.csv
  - results/tables/jscc_all_cbr_mc_summary.csv

Sử dụng:
    python src/plot_results.py --input_dir results/tables --output_dir results/figures
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Màu sắc cho từng model
COLORS = {
    "Deep JSCC cbr_1_12": "#1565C0",  # xanh đậm
    "Deep JSCC cbr_1_6": "#2196F3",   # xanh
    "Deep JSCC cbr_1_4": "#64B5F6",   # xanh nhạt
    "JPEG+BPSK": "#F44336",           # đỏ
    "JPEG+Rep3+BPSK": "#FF9800",      # cam
}

MARKERS = {
    "Deep JSCC cbr_1_12": "o",
    "Deep JSCC cbr_1_6": "s",
    "Deep JSCC cbr_1_4": "^",
    "JPEG+BPSK": "D",
    "JPEG+Rep3+BPSK": "v",
}

CBR_LABELS = {
    "cbr_1_12": "CBR=1/12",
    "cbr_1_6": "CBR=1/6",
    "cbr_1_4": "CBR=1/4",
}


def set_plot_style():
    """Thiết lập style chung cho tất cả plot."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.grid": True,
        "grid.color": "#DEE2E6",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def load_jscc_summaries(input_dir: str) -> Dict[str, pd.DataFrame]:
    """Đọc tất cả file jscc_*_mc_summary.csv."""
    summaries = {}
    for tag in ["cbr_1_12", "cbr_1_6", "cbr_1_4"]:
        path = os.path.join(input_dir, f"jscc_{tag}_mc_summary.csv")
        if os.path.exists(path):
            summaries[tag] = pd.read_csv(path)
            logger.info(f"Đọc: {path}")
        else:
            logger.warning(f"Không tìm thấy: {path}")
    return summaries


def load_baseline_summary(input_dir: str) -> Optional[pd.DataFrame]:
    """Đọc baseline_mc_summary.csv."""
    path = os.path.join(input_dir, "baseline_mc_summary.csv")
    if os.path.exists(path):
        logger.info(f"Đọc: {path}")
        return pd.read_csv(path)
    logger.warning(f"Không tìm thấy: {path}")
    return None


def plot_psnr_vs_snr(
    jscc_summaries: Dict[str, pd.DataFrame],
    baseline_df: Optional[pd.DataFrame],
    output_path: str,
):
    """Vẽ PSNR vs SNR với error bar."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ Deep JSCC cho từng CBR
    for tag, df in jscc_summaries.items():
        label = f"Deep JSCC {CBR_LABELS.get(tag, tag)}"
        color = COLORS.get(f"Deep JSCC {tag}", "#2196F3")
        marker = MARKERS.get(f"Deep JSCC {tag}", "o")
        snr_arr = df["snr_db"].values
        psnr_mean = df["psnr_mean"].values
        psnr_std = df["psnr_std"].values
        ax.errorbar(
            snr_arr, psnr_mean, yerr=psnr_std,
            label=label, color=color, marker=marker,
            linewidth=2, markersize=7, capsize=5,
            linestyle="-",
        )

    # Vẽ baseline
    if baseline_df is not None:
        for model_name in baseline_df["model"].unique():
            subset = baseline_df[baseline_df["model"] == model_name]
            # Lấy quality trung bình hoặc quality tốt nhất
            # Group by SNR, lấy quality cho PSNR cao nhất
            best_records = []
            for snr in subset["snr_db"].unique():
                snr_sub = subset[subset["snr_db"] == snr]
                best_row = snr_sub.loc[snr_sub["psnr_mean"].idxmax()]
                best_records.append(best_row)
            best_df = pd.DataFrame(best_records).sort_values("snr_db")

            color = COLORS.get(model_name, "#9E9E9E")
            marker = MARKERS.get(model_name, "x")
            ax.errorbar(
                best_df["snr_db"].values,
                best_df["psnr_mean"].values,
                yerr=best_df["psnr_std"].values,
                label=f"{model_name} (best Q)",
                color=color, marker=marker,
                linewidth=1.5, markersize=7, capsize=5,
                linestyle="--",
            )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs SNR — Deep JSCC vs Baseline (AWGN Channel)")
    ax.legend(loc="upper left")
    ax.set_xticks([0, 5, 10, 15, 20])

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_ssim_vs_snr(
    jscc_summaries: Dict[str, pd.DataFrame],
    baseline_df: Optional[pd.DataFrame],
    output_path: str,
):
    """Vẽ SSIM vs SNR với error bar."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for tag, df in jscc_summaries.items():
        label = f"Deep JSCC {CBR_LABELS.get(tag, tag)}"
        color = COLORS.get(f"Deep JSCC {tag}", "#2196F3")
        marker = MARKERS.get(f"Deep JSCC {tag}", "o")
        ax.errorbar(
            df["snr_db"].values, df["ssim_mean"].values,
            yerr=df["ssim_std"].values,
            label=label, color=color, marker=marker,
            linewidth=2, markersize=7, capsize=5,
        )

    if baseline_df is not None:
        for model_name in baseline_df["model"].unique():
            subset = baseline_df[baseline_df["model"] == model_name]
            best_records = []
            for snr in subset["snr_db"].unique():
                snr_sub = subset[subset["snr_db"] == snr]
                best_row = snr_sub.loc[snr_sub["ssim_mean"].idxmax()]
                best_records.append(best_row)
            best_df = pd.DataFrame(best_records).sort_values("snr_db")

            color = COLORS.get(model_name, "#9E9E9E")
            marker = MARKERS.get(model_name, "x")
            ax.errorbar(
                best_df["snr_db"].values, best_df["ssim_mean"].values,
                yerr=best_df["ssim_std"].values,
                label=f"{model_name} (best Q)",
                color=color, marker=marker,
                linewidth=1.5, markersize=7, capsize=5, linestyle="--",
            )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SSIM")
    ax.set_title("SSIM vs SNR — Deep JSCC vs Baseline (AWGN Channel)")
    ax.legend(loc="upper left")
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_cbr_comparison(
    jscc_summaries: Dict[str, pd.DataFrame],
    output_path: str,
):
    """Vẽ so sánh PSNR theo CBR."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    for tag, df in jscc_summaries.items():
        label = CBR_LABELS.get(tag, tag)
        color = COLORS.get(f"Deep JSCC {tag}", "#2196F3")
        marker = MARKERS.get(f"Deep JSCC {tag}", "o")
        ax.plot(
            df["snr_db"].values, df["psnr_mean"].values,
            label=label, color=color, marker=marker,
            linewidth=2, markersize=8,
        )
        # Vùng ± std
        ax.fill_between(
            df["snr_db"].values,
            df["psnr_mean"].values - df["psnr_std"].values,
            df["psnr_mean"].values + df["psnr_std"].values,
            alpha=0.15, color=color,
        )

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Ảnh hưởng của CBR đến PSNR — Deep JSCC trên AWGN")
    ax.legend(title="CBR", loc="upper left")
    ax.set_xticks([0, 5, 10, 15, 20])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def plot_reconstruction_grid_dummy(output_path: str, image_size: int = 64):
    """
    Tạo reconstruction grid placeholder khi không có ảnh thực.
    Dùng ảnh ngẫu nhiên để minh hoạ layout.
    """
    set_plot_style()
    np.random.seed(42)
    snr_list = [0, 5, 10, 15, 20]
    n_images = 3

    fig, axes = plt.subplots(
        n_images, len(snr_list) + 1,
        figsize=(3 * (len(snr_list) + 1), 3 * n_images)
    )
    fig.suptitle("Reconstruction Grid — Deep JSCC qua AWGN", fontsize=14, y=1.02)

    for i in range(n_images):
        orig = np.random.rand(image_size, image_size, 3).astype(np.float32)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis("off")
        axes[i, 0].set_ylabel(f"Image {i+1}", rotation=90, labelpad=10)

        for j, snr in enumerate(snr_list):
            snr_linear = 10 ** (snr / 10)
            noise = np.random.randn(*orig.shape) * np.sqrt(1 / (2 * snr_linear))
            recon = np.clip(orig + noise * 0.1, 0, 1)
            axes[i, j + 1].imshow(recon)
            if i == 0:
                axes[i, j + 1].set_title(f"SNR={snr} dB")
            axes[i, j + 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {output_path}")


def create_final_table(
    jscc_summaries: Dict[str, pd.DataFrame],
    baseline_df: Optional[pd.DataFrame],
    output_path: str,
):
    """Tạo bảng tổng hợp final_psnr_ssim_table.csv."""
    rows = []

    for tag, df in jscc_summaries.items():
        cbr_label = CBR_LABELS.get(tag, tag)
        for _, row in df.iterrows():
            rows.append({
                "model": f"Deep JSCC",
                "cbr_label": cbr_label,
                "snr_db": row["snr_db"],
                "psnr_mean±std": f"{row['psnr_mean']:.2f}±{row['psnr_std']:.2f}",
                "ssim_mean±std": f"{row['ssim_mean']:.4f}±{row['ssim_std']:.4f}",
                "mse_mean±std": f"{row['mse_mean']:.6f}±{row['mse_std']:.6f}",
                "failure_rate": row.get("failure_rate_mean", 0),
            })

    if baseline_df is not None:
        for model_name in baseline_df["model"].unique():
            subset = baseline_df[baseline_df["model"] == model_name]
            for snr in sorted(subset["snr_db"].unique()):
                snr_sub = subset[subset["snr_db"] == snr]
                best_row = snr_sub.loc[snr_sub["psnr_mean"].idxmax()]
                rows.append({
                    "model": model_name,
                    "cbr_label": f"CBR≈{best_row['cbr']:.3f}",
                    "snr_db": snr,
                    "psnr_mean±std": f"{best_row['psnr_mean']:.2f}±{best_row['psnr_std']:.2f}",
                    "ssim_mean±std": f"{best_row['ssim_mean']:.4f}±{best_row['ssim_std']:.4f}",
                    "mse_mean±std": f"{best_row['mse_mean']:.6f}±{best_row['mse_std']:.6f}",
                    "failure_rate": best_row.get("failure_rate", 0),
                })

    final_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")


def create_simulation_settings_table(output_path: str):
    """Tạo bảng thông số mô phỏng."""
    settings = pd.DataFrame([
        {"Tham số": "Dataset", "Giá trị": "CIFAR-10"},
        {"Tham số": "Image size", "Giá trị": "64×64 pixels"},
        {"Tham số": "Channel", "Giá trị": "AWGN (Additive White Gaussian Noise)"},
        {"Tham số": "SNR test", "Giá trị": "0, 5, 10, 15, 20 dB"},
        {"Tham số": "Monte Carlo K", "Giá trị": "20 lần/SNR"},
        {"Tham số": "CBR", "Giá trị": "1/12, 1/6, 1/4"},
        {"Tham số": "Train SNR", "Giá trị": "10 dB"},
        {"Tham số": "Loss function", "Giá trị": "MSE"},
        {"Tham số": "Optimizer", "Giá trị": "Adam (lr=1e-3)"},
        {"Tham số": "Epochs", "Giá trị": "100"},
        {"Tham số": "Batch size", "Giá trị": "64"},
        {"Tham số": "Metrics", "Giá trị": "MSE, PSNR, SSIM"},
        {"Tham số": "Baseline 1", "Giá trị": "JPEG + BPSK"},
        {"Tham số": "Baseline 2", "Giá trị": "JPEG + Repetition Code (×3) + BPSK"},
    ])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    settings.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path}")


def merge_all_cbr_summaries(
    jscc_summaries: Dict[str, pd.DataFrame],
    output_path: str,
):
    """Gộp tất cả CBR summaries thành một file."""
    dfs = []
    for tag, df in jscc_summaries.items():
        df = df.copy()
        df["cbr_tag"] = tag
        df["cbr_label"] = CBR_LABELS.get(tag, tag)
        dfs.append(df)
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sinh hình vẽ và bảng biểu cho báo cáo")
    parser.add_argument("--input_dir", type=str, default="results/tables")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    parser.add_argument("--image_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.input_dir, exist_ok=True)

    logger.info("=== Sinh hình vẽ và bảng biểu ===")

    # Đọc dữ liệu
    jscc_summaries = load_jscc_summaries(args.input_dir)
    baseline_df = load_baseline_summary(args.input_dir)

    if not jscc_summaries:
        logger.warning("Không tìm thấy JSCC summary. Vẽ placeholder...")

    # 1. PSNR vs SNR
    plot_psnr_vs_snr(
        jscc_summaries, baseline_df,
        os.path.join(args.output_dir, "final_psnr_vs_snr.png")
    )

    # 2. SSIM vs SNR
    plot_ssim_vs_snr(
        jscc_summaries, baseline_df,
        os.path.join(args.output_dir, "final_ssim_vs_snr.png")
    )

    # 3. CBR comparison
    if jscc_summaries:
        plot_cbr_comparison(
            jscc_summaries,
            os.path.join(args.output_dir, "jscc_cbr_comparison_psnr.png")
        )

    # 4. Reconstruction grid
    plot_reconstruction_grid_dummy(
        os.path.join(args.output_dir, "reconstruction_grid.png"),
        image_size=args.image_size,
    )

    # 5. Tạo bảng tổng hợp
    create_final_table(
        jscc_summaries, baseline_df,
        os.path.join(args.input_dir, "final_psnr_ssim_table.csv")
    )

    # 6. Bảng thông số mô phỏng
    create_simulation_settings_table(
        os.path.join(args.input_dir, "final_simulation_settings.csv")
    )

    # 7. Gộp tất cả CBR
    merge_all_cbr_summaries(
        jscc_summaries,
        os.path.join(args.input_dir, "jscc_all_cbr_mc_summary.csv")
    )

    logger.info("✅ Hoàn tất sinh hình vẽ và bảng biểu!")
    logger.info(f"Hình lưu tại: {args.output_dir}/")
    logger.info(f"Bảng lưu tại: {args.input_dir}/")


if __name__ == "__main__":
    main()
