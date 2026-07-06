"""
Baseline Evaluation: JPEG + BPSK và JPEG + Repetition Code + BPSK
------------------------------------------------------------------
Mô phỏng truyền ảnh dùng phương pháp truyền thống:
  1. JPEG + BPSK (không có mã hoá kênh)
  2. JPEG + Repetition Code + BPSK

Pipeline:
  1. Nén ảnh bằng JPEG (quality Q)
  2. Chuyển bytes JPEG thành bits BPSK (+1/-1)
  3. Thêm nhiễu AWGN vào tín hiệu BPSK
  4. Giải mã BPSK (hard decision)
  5. Giải nén JPEG để lấy ảnh khôi phục
  6. Tính MSE, PSNR, SSIM

Lưu ý:
  - Khi SNR thấp, tỷ lệ bit lỗi (BER) tăng cao → JPEG decode thất bại → failure_rate tăng
  - Repetition code (factor=3) lặp lại mỗi bit 3 lần, dùng majority voting để decode
  - CBR của baseline được tính dựa trên kích thước JPEG và repetition factor
"""

import argparse
import io
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_test_loader_only
from src.metrics.image_metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Chuyển tensor [C, H, W] trong [0,1] sang PIL Image."""
    arr = (img_tensor.clamp(0, 1).numpy() * 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    return Image.fromarray(arr)


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """Chuyển PIL Image sang tensor [C, H, W] trong [0,1]."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    return torch.from_numpy(arr).permute(2, 0, 1)


def jpeg_encode(pil_img: Image.Image, quality: int) -> bytes:
    """Nén ảnh JPEG và trả về bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Chuyển bytes sang mảng bit nhị phân (0/1)."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits.astype(np.float32)


def bits_to_bpsk(bits: np.ndarray) -> np.ndarray:
    """Chuyển bits {0,1} sang symbols BPSK {-1, +1}."""
    return 2.0 * bits - 1.0  # 0 -> -1, 1 -> +1


def add_awgn(symbols: np.ndarray, snr_db: float) -> np.ndarray:
    """Thêm nhiễu AWGN vào symbols BPSK."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_std = np.sqrt(1.0 / (2.0 * snr_linear))
    noise = np.random.randn(*symbols.shape) * noise_std
    return symbols + noise


def bpsk_hard_decision(received: np.ndarray) -> np.ndarray:
    """Hard decision decoder cho BPSK. Trả về bits {0,1}."""
    return (received > 0).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Chuyển mảng bit nhị phân (0/1) sang bytes."""
    # Padding nếu cần
    n = len(bits)
    if n % 8 != 0:
        bits = np.concatenate([bits, np.zeros(8 - n % 8, dtype=np.uint8)])
    return np.packbits(bits.astype(np.uint8)).tobytes()


def jpeg_decode_safe(data: bytes, target_size: int) -> Tuple[Image.Image, bool]:
    """
    Giải nén JPEG từ bytes. Trả về (PIL Image, success flag).
    Nếu fail, trả về ảnh đen.
    """
    try:
        buf = io.BytesIO(data)
        img = Image.open(buf)
        img.load()
        img = img.resize((target_size, target_size), Image.BILINEAR)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img, True
    except Exception:
        # Failure: trả về ảnh đen
        return Image.new("RGB", (target_size, target_size), color=(0, 0, 0)), False


def apply_repetition_code(bits: np.ndarray, factor: int) -> np.ndarray:
    """Lặp lại mỗi bit `factor` lần (repetition encoding)."""
    return np.repeat(bits, factor)


def decode_repetition_code(received_bits: np.ndarray, factor: int) -> np.ndarray:
    """Majority voting để giải mã repetition code."""
    n_original = len(received_bits) // factor
    reshaped = received_bits[:n_original * factor].reshape(n_original, factor)
    majority = (reshaped.sum(axis=1) > factor / 2).astype(np.uint8)
    return majority


def compute_bitrate(jpeg_bytes: bytes, img_size: int, rep_factor: int = 1) -> float:
    """
    Tính CBR (Channel Bandwidth Ratio) của baseline.

    CBR = (n_bits_transmitted) / (C * H * W)
    """
    n_bits_jpeg = len(jpeg_bytes) * 8
    n_bits_tx = n_bits_jpeg * rep_factor  # Sau repetition code
    n_pixels_total = 3 * img_size * img_size  # C*H*W
    return n_bits_tx / n_pixels_total


@torch.no_grad()
def eval_jpeg_bpsk(
    test_loader,
    snr_db_list: List[float],
    jpeg_quality_list: List[int],
    image_size: int,
    rep_factor: int = 1,
) -> pd.DataFrame:
    """
    Đánh giá baseline JPEG + BPSK (hoặc + Repetition Code).

    Args:
        test_loader: DataLoader chứa test images.
        snr_db_list: Danh sách SNR (dB) để test.
        jpeg_quality_list: Danh sách JPEG quality factors.
        image_size: Kích thước ảnh.
        rep_factor: Repetition factor (1 = no repetition, 3 = triple repetition).

    Returns:
        DataFrame với cột: model, jpeg_quality, rep_factor, cbr, snr_db,
                            psnr_mean, psnr_std, ssim_mean, ssim_std,
                            mse_mean, mse_std, failure_rate.
    """
    baseline_name = "JPEG+BPSK" if rep_factor == 1 else f"JPEG+Rep{rep_factor}+BPSK"
    records = []

    for snr_db in snr_db_list:
        for quality in jpeg_quality_list:
            psnr_list, ssim_list, mse_list = [], [], []
            failures = 0
            total = 0
            cbr_list = []

            for imgs_batch, _ in test_loader:
                for i in range(imgs_batch.shape[0]):
                    img_tensor = imgs_batch[i]  # [C, H, W]
                    pil_img = tensor_to_pil(img_tensor)

                    # 1. JPEG encoding
                    jpeg_bytes = jpeg_encode(pil_img, quality=quality)
                    cbr = compute_bitrate(jpeg_bytes, image_size, rep_factor)
                    cbr_list.append(cbr)

                    # 2. Chuyển sang bits
                    bits = bytes_to_bits(jpeg_bytes)

                    # 3. Repetition encoding
                    if rep_factor > 1:
                        bits_tx = apply_repetition_code(bits, rep_factor)
                    else:
                        bits_tx = bits

                    # 4. BPSK modulation
                    symbols = bits_to_bpsk(bits_tx)

                    # 5. AWGN channel
                    received = add_awgn(symbols, snr_db)

                    # 6. Hard decision
                    rx_bits = bpsk_hard_decision(received)

                    # 7. Repetition decoding
                    if rep_factor > 1:
                        rx_bits = decode_repetition_code(rx_bits, rep_factor)

                    # 8. Chuyển lại bytes
                    rx_bytes = bits_to_bytes(rx_bits)

                    # 9. JPEG decode
                    recon_pil, success = jpeg_decode_safe(rx_bytes, image_size)
                    if not success:
                        failures += 1

                    # 10. Tính metrics
                    recon_tensor = pil_to_tensor(recon_pil).unsqueeze(0)
                    orig_tensor = img_tensor.unsqueeze(0)
                    mse_val, psnr_val, ssim_val = compute_all_metrics(orig_tensor, recon_tensor)

                    psnr_list.append(psnr_val)
                    ssim_list.append(ssim_val)
                    mse_list.append(mse_val)
                    total += 1

            failure_rate = failures / max(total, 1)
            avg_cbr = np.mean(cbr_list) if cbr_list else 0.0

            records.append({
                "model": baseline_name,
                "jpeg_quality": quality,
                "rep_factor": rep_factor,
                "cbr": avg_cbr,
                "snr_db": snr_db,
                "psnr_mean": np.mean(psnr_list),
                "psnr_std": np.std(psnr_list),
                "ssim_mean": np.mean(ssim_list),
                "ssim_std": np.std(ssim_list),
                "mse_mean": np.mean(mse_list),
                "mse_std": np.std(mse_list),
                "failure_rate": failure_rate,
                "n_images": total,
            })
            logger.info(
                f"{baseline_name} | Q={quality:2d} | SNR={snr_db:3.0f} dB | "
                f"PSNR={np.mean(psnr_list):.2f} dB | SSIM={np.mean(ssim_list):.4f} | "
                f"CBR={avg_cbr:.4f} | FailRate={failure_rate:.2%}"
            )

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá baseline JPEG+BPSK và JPEG+Rep+BPSK")
    parser.add_argument("--config", type=str, default="configs/cbr_1_6.yaml")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="results/tables")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    channel_cfg = config.get("channel", {})
    baseline_cfg = config.get("baseline", {})

    snr_list = channel_cfg.get("test_snr_db", [0, 5, 10, 15, 20])
    jpeg_quality_list = baseline_cfg.get("jpeg_quality_list", [5, 10, 20, 30, 50])
    rep_factor = baseline_cfg.get("repetition_factor", 3)
    image_size = config.get("image_size", 64)

    logger.info(f"SNR list: {snr_list}")
    logger.info(f"JPEG quality list: {jpeg_quality_list}")
    logger.info(f"Repetition factor: {rep_factor}")

    # DataLoader (chỉ cần subset nhỏ vì baseline chậm)
    test_loader = get_test_loader_only(config, data_root=args.data_root)

    # Đánh giá JPEG + BPSK (no repetition)
    logger.info("\n=== Đánh giá JPEG + BPSK ===")
    df_bpsk = eval_jpeg_bpsk(
        test_loader, snr_list, jpeg_quality_list, image_size, rep_factor=1
    )

    # Đánh giá JPEG + Repetition + BPSK
    logger.info(f"\n=== Đánh giá JPEG + Rep{rep_factor} + BPSK ===")
    df_rep = eval_jpeg_bpsk(
        test_loader, snr_list, jpeg_quality_list, image_size, rep_factor=rep_factor
    )

    # Gộp và lưu
    df_all = pd.concat([df_bpsk, df_rep], ignore_index=True)
    raw_path = os.path.join(args.output_dir, "baseline_mc_raw.csv")
    df_all.to_csv(raw_path, index=False)
    logger.info(f"Raw baseline results: {raw_path}")

    # Summary: best quality cho mỗi SNR
    summary_path = os.path.join(args.output_dir, "baseline_mc_summary.csv")
    df_all.to_csv(summary_path, index=False)
    logger.info(f"Baseline summary: {summary_path}")

    # Bảng bitrate
    bitrate_path = os.path.join(args.output_dir, "baseline_bitrate_table.csv")
    df_bitrate = df_all[["model", "jpeg_quality", "rep_factor", "cbr"]].drop_duplicates()
    df_bitrate.to_csv(bitrate_path, index=False)
    logger.info(f"Bitrate table: {bitrate_path}")

    logger.info("✅ Baseline evaluation hoàn tất!")


if __name__ == "__main__":
    main()
