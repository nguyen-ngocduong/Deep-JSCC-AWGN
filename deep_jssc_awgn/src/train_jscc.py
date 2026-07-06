"""
Training Script cho Deep JSCC
------------------------------
Huấn luyện mô hình Deep JSCC với MSE loss trên kênh AWGN.

Sử dụng:
    python src/train_jscc.py --config configs/cbr_1_6.yaml

Checkpoint được lưu tại: results/checkpoints/jscc_<tag>_best.pt
Loss curve được lưu tại: results/figures/loss_curve_<tag>.png
Log CSV được lưu tại:    results/tables/train_<tag>.csv
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# Thêm root project vào sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_dataloaders
from src.models.deep_jscc import build_model
from src.metrics.image_metrics import compute_psnr, compute_ssim

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/logs/app.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_cbr_tag(config: dict) -> str:
    """Tạo tag tên file từ config CBR."""
    jscc_cfg = config.get("jscc", {})
    cbr = jscc_cfg.get("cbr", 1 / 6)
    if abs(cbr - 1 / 12) < 0.01:
        return "cbr_1_12"
    elif abs(cbr - 1 / 6) < 0.01:
        return "cbr_1_6"
    elif abs(cbr - 1 / 4) < 0.01:
        return "cbr_1_4"
    else:
        return f"cbr_{cbr:.4f}".replace(".", "_")


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Chạy một epoch training. Trả về loss trung bình."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        optimizer.zero_grad()
        recon = model(imgs)
        loss = criterion(recon, imgs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Chạy validation. Trả về dict chứa val_mse, val_psnr, val_ssim.
    """
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        recon = model(imgs)
        loss = criterion(recon, imgs)
        total_mse += loss.item()
        total_psnr += compute_psnr(imgs.cpu(), recon.cpu())
        total_ssim += compute_ssim(imgs.cpu(), recon.cpu())
        n_batches += 1

    return {
        "val_mse": total_mse / max(n_batches, 1),
        "val_psnr": total_psnr / max(n_batches, 1),
        "val_ssim": total_ssim / max(n_batches, 1),
    }


def save_loss_curve(history: dict, save_path: str):
    """Vẽ và lưu loss curve train/val."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train MSE", color="#2196F3")
    axes[0].plot(epochs, history["val_mse"], label="Val MSE", color="#F44336")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Train / Validation MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_psnr"], label="Val PSNR", color="#4CAF50")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Validation PSNR")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["val_ssim"], label="Val SSIM", color="#FF9800")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].set_title("Validation SSIM")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss curve lưu tại: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Deep JSCC trên AWGN")
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
        help="Thư mục raw dataset",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Đường dẫn checkpoint để resume",
    )
    args = parser.parse_args()

    # Tạo thư mục cần thiết
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    config = load_config(args.config)
    tag = get_cbr_tag(config)
    train_cfg = config.get("train", {})
    channel_cfg = config.get("channel", {})

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Thiết bị: {device}")

    # Fix random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build model
    model = build_model(config).to(device)
    logger.info(f"Model: {model.__class__.__name__}, CBR={model.get_cbr():.4f}")
    logger.info(f"Số tham số: {model.count_parameters():,}")

    # DataLoader
    train_loader, val_loader, _ = get_dataloaders(
        config, data_root=args.data_root, augment=True
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Loss và Optimizer
    criterion = nn.MSELoss()
    lr = train_cfg.get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    epochs = train_cfg.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume nếu có
    start_epoch = 1
    best_val_mse = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_mse = ckpt.get("best_val_mse", float("inf"))
        logger.info(f"Resume từ epoch {start_epoch}, best_val_mse={best_val_mse:.6f}")

    # Training loop
    history = {
        "epoch": [],
        "train_loss": [],
        "val_mse": [],
        "val_psnr": [],
        "val_ssim": [],
        "lr": [],
    }

    ckpt_path = f"results/checkpoints/jscc_{tag}_snr{int(channel_cfg.get('train_snr_db', 10))}_best.pt"
    logger.info(f"Checkpoint sẽ lưu tại: {ckpt_path}")
    logger.info(f"Bắt đầu training từ epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_mse"].append(val_metrics["val_mse"])
        history["val_psnr"].append(val_metrics["val_psnr"])
        history["val_ssim"].append(val_metrics["val_ssim"])
        history["lr"].append(current_lr)

        is_best = val_metrics["val_mse"] < best_val_mse
        if is_best:
            best_val_mse = val_metrics["val_mse"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mse": best_val_mse,
                    "config": config,
                    "tag": tag,
                },
                ckpt_path,
            )

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Val MSE: {val_metrics['val_mse']:.6f} | "
            f"Val PSNR: {val_metrics['val_psnr']:.2f} dB | "
            f"Val SSIM: {val_metrics['val_ssim']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{'✅ BEST' if is_best else ''} "
            f"[{elapsed:.1f}s]"
        )

    # Lưu log CSV
    log_csv_path = f"results/tables/train_{tag}_snr{int(channel_cfg.get('train_snr_db', 10))}.csv"
    pd.DataFrame(history).to_csv(log_csv_path, index=False)
    logger.info(f"Training log lưu tại: {log_csv_path}")

    # Vẽ loss curve
    loss_curve_path = f"results/figures/loss_curve_{tag}.png"
    save_loss_curve(history, loss_curve_path)

    logger.info(f"✅ Training hoàn tất! Best val MSE: {best_val_mse:.6f}")
    logger.info(f"Best checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
