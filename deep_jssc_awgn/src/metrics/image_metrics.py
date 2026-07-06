"""
Image Quality Metrics Module
-----------------------------
Tính MSE, PSNR và SSIM cho ảnh đã khôi phục.
"""

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage
from typing import Tuple


def compute_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Tính Mean Squared Error giữa ảnh gốc và ảnh khôi phục.

    Args:
        original: Tensor ảnh gốc [B, C, H, W], giá trị trong [0, 1].
        reconstructed: Tensor ảnh khôi phục [B, C, H, W].

    Returns:
        Giá trị MSE trung bình trên toàn batch (float).
    """
    with torch.no_grad():
        mse = torch.mean((original - reconstructed) ** 2)
    return mse.item()


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor,
                 max_val: float = 1.0) -> float:
    """
    Tính Peak Signal-to-Noise Ratio (PSNR) tính bằng dB.

    PSNR = 10 * log10(max_val^2 / MSE)

    Args:
        original: Tensor ảnh gốc [B, C, H, W], giá trị trong [0, 1].
        reconstructed: Tensor ảnh khôi phục [B, C, H, W].
        max_val: Giá trị tối đa của pixel (mặc định 1.0).

    Returns:
        Giá trị PSNR trung bình (dB).
    """
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        return float("inf")
    psnr = 10.0 * np.log10(max_val ** 2 / mse)
    return float(psnr)


def compute_ssim(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Tính Structural Similarity Index (SSIM) dùng skimage.

    Args:
        original: Tensor ảnh gốc [B, C, H, W], giá trị trong [0, 1].
        reconstructed: Tensor ảnh khôi phục [B, C, H, W].

    Returns:
        Giá trị SSIM trung bình trên toàn batch (float).
    """
    orig_np = original.detach().cpu().clamp(0, 1).numpy()
    recon_np = reconstructed.detach().cpu().clamp(0, 1).numpy()

    ssim_scores = []
    for i in range(orig_np.shape[0]):
        # Chuyển từ (C, H, W) -> (H, W, C) cho skimage
        orig_img = np.transpose(orig_np[i], (1, 2, 0))
        recon_img = np.transpose(recon_np[i], (1, 2, 0))

        score = ssim_skimage(
            orig_img,
            recon_img,
            data_range=1.0,
            channel_axis=2,
        )
        ssim_scores.append(score)

    return float(np.mean(ssim_scores))


def compute_all_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_val: float = 1.0
) -> Tuple[float, float, float]:
    """
    Tính đồng thời MSE, PSNR và SSIM.

    Args:
        original: Tensor ảnh gốc [B, C, H, W].
        reconstructed: Tensor ảnh khôi phục [B, C, H, W].
        max_val: Giá trị pixel tối đa.

    Returns:
        Tuple (mse, psnr_db, ssim) là các giá trị float.
    """
    mse = compute_mse(original, reconstructed)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(max_val ** 2 / mse)
    ssim_val = compute_ssim(original, reconstructed)
    return float(mse), float(psnr), ssim_val
