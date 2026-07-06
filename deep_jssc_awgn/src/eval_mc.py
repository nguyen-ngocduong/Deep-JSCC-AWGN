"""
Monte Carlo Evaluation Script
------------------------------
Đánh giá mô hình Deep JSCC bằng phương pháp Monte Carlo:
- Chạy K lần inference cho mỗi giá trị SNR
- Tính mean ± std của PSNR, SSIM, MSE
- Báo cáo failure_rate (số lần model cho PSNR < ngưỡng)

Sử dụng:
    python src/eval_mc.py \
        --config configs/cbr_1_6.yaml \
        --ckpt results/checkpoints/jscc_cbr_1_6_snr10_best.pt

Đầu ra:
    results/tables/jscc_<tag>_mc_raw.csv
    results/tables/jscc_<tag>_mc_summary.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_test_loader_only
from src.models.deep_jscc import build_model
from src.metrics.image_metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PSNR_FAILURE_THRESHOLD = 5.0  # dB – dưới ngưỡng này tính là failure


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cbr_tag(config: dict) -> str:
    cbr = config.get("jscc", {}).get("cbr", 1 / 6)
    if abs(cbr - 1 / 12) < 0.01:
        return "cbr_1_12"
    elif abs(cbr - 1 / 6) < 0.01:
        return "cbr_1_6"
    elif abs(cbr - 1 / 4) < 0.01:
        return "cbr_1_4"
    else:
        return f"cbr_{cbr:.4f}".replace(".", "_")


@torch.no_grad()
def monte_carlo_eval_snr(
    model: torch.nn.Module,
    loader,
    snr_db: float,
    n_runs: int,
    device: torch.device,
) -> list:
    """
    Chạy Monte Carlo K lần cho một giá trị SNR.

    Mỗi run: set SNR -> chạy toàn bộ test set -> tính metrics.

    Returns:
        List of dicts, mỗi dict là kết quả một run.
    """
    model.eval()
    records = []

    for run_idx in range(n_runs):
        run_mse_list = []
        run_psnr_list = []
        run_ssim_list = []
        failures = 0
        total = 0

        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            recon = model(imgs, snr_db=snr_db)
            mse, psnr, ssim = compute_all_metrics(imgs.cpu(), recon.cpu())
            run_mse_list.append(mse)
            run_psnr_list.append(psnr)
            run_ssim_list.append(ssim)
            failures += int(psnr < PSNR_FAILURE_THRESHOLD)
            total += 1

        run_mse = np.mean(run_mse_list)
        run_psnr = np.mean(run_psnr_list)
        run_ssim = np.mean(run_ssim_list)
        failure_rate = failures / max(total, 1)

        records.append({
            "snr_db": snr_db,
            "run": run_idx,
            "mse": run_mse,
            "psnr": run_psnr,
            "ssim": run_ssim,
            "failure_rate": failure_rate,
        })
        logger.debug(
            f"  SNR={snr_db:3.0f} dB | Run {run_idx+1:2d}/{n_runs} | "
            f"PSNR={run_psnr:.2f} dB | SSIM={run_ssim:.4f} | MSE={run_mse:.6f}"
        )

    return records


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation của Deep JSCC")
    parser.add_argument("--config", type=str, default="configs/cbr_1_6.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Đường dẫn checkpoint .pt")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=None,
        help="Số Monte Carlo runs (override config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/tables",
        help="Thư mục lưu kết quả CSV",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    tag = get_cbr_tag(config)
    channel_cfg = config.get("channel", {})
    n_runs = args.n_runs or channel_cfg.get("monte_carlo_runs", 20)
    snr_list = channel_cfg.get("test_snr_db", [0, 5, 10, 15, 20])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị: {device}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info(f"Monte Carlo runs/SNR: {n_runs}")
    logger.info(f"SNR list: {snr_list}")

    # Load model
    model = build_model(config).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Model loaded: CBR={model.get_cbr():.4f}")

    # DataLoader
    test_loader = get_test_loader_only(config, data_root=args.data_root)
    logger.info(f"Test batches: {len(test_loader)}")

    # Monte Carlo evaluation
    all_records = []
    for snr_db in snr_list:
        logger.info(f"Đang eval SNR = {snr_db} dB ({n_runs} runs)...")
        records = monte_carlo_eval_snr(model, test_loader, float(snr_db), n_runs, device)
        all_records.extend(records)
        snr_psnr = np.mean([r["psnr"] for r in records])
        snr_ssim = np.mean([r["ssim"] for r in records])
        snr_mse = np.mean([r["mse"] for r in records])
        logger.info(
            f"  SNR={snr_db} dB -> PSNR={snr_psnr:.2f} dB, SSIM={snr_ssim:.4f}, MSE={snr_mse:.6f}"
        )

    # Lưu raw CSV
    raw_df = pd.DataFrame(all_records)
    raw_path = os.path.join(args.output_dir, f"jscc_{tag}_mc_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    logger.info(f"Raw results: {raw_path}")

    # Tạo summary (mean ± std)
    summary_records = []
    for snr_db in snr_list:
        subset = raw_df[raw_df["snr_db"] == snr_db]
        summary_records.append({
            "model": "Deep JSCC",
            "cbr": config.get("jscc", {}).get("cbr", "N/A"),
            "snr_db": snr_db,
            "psnr_mean": subset["psnr"].mean(),
            "psnr_std": subset["psnr"].std(),
            "ssim_mean": subset["ssim"].mean(),
            "ssim_std": subset["ssim"].std(),
            "mse_mean": subset["mse"].mean(),
            "mse_std": subset["mse"].std(),
            "failure_rate_mean": subset["failure_rate"].mean(),
            "n_runs": n_runs,
        })

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(args.output_dir, f"jscc_{tag}_mc_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary results: {summary_path}")

    # In bảng kết quả
    logger.info("\n=== Bảng tóm tắt kết quả Monte Carlo ===")
    for _, row in summary_df.iterrows():
        logger.info(
            f"SNR={row['snr_db']:3.0f} dB | "
            f"PSNR={row['psnr_mean']:.2f}±{row['psnr_std']:.2f} dB | "
            f"SSIM={row['ssim_mean']:.4f}±{row['ssim_std']:.4f} | "
            f"MSE={row['mse_mean']:.6f}±{row['mse_std']:.6f}"
        )

    logger.info("✅ Monte Carlo evaluation hoàn tất!")


if __name__ == "__main__":
    main()
