"""
Unit Tests cho Image Metrics
-----------------------------
Test: MSE, PSNR, SSIM tính đúng theo spec.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.image_metrics import (
    compute_mse,
    compute_psnr,
    compute_ssim,
    compute_all_metrics,
)


class TestMSE:
    """Test compute_mse."""

    def test_identical_images_zero_mse(self):
        """Ảnh gốc và khôi phục giống nhau -> MSE = 0."""
        x = torch.rand(4, 3, 64, 64)
        mse = compute_mse(x, x)
        assert mse == pytest.approx(0.0, abs=1e-7), f"MSE phải = 0 khi ảnh giống nhau: {mse}"

    def test_mse_nonnegative(self):
        """MSE luôn >= 0."""
        x = torch.rand(4, 3, 64, 64)
        y = torch.rand(4, 3, 64, 64)
        assert compute_mse(x, y) >= 0

    def test_mse_symmetry(self):
        """MSE(x, y) == MSE(y, x)."""
        x = torch.rand(4, 3, 64, 64)
        y = torch.rand(4, 3, 64, 64)
        assert compute_mse(x, y) == pytest.approx(compute_mse(y, x), rel=1e-6)

    def test_mse_scalar_offset(self):
        """MSE với ảnh có offset cố định."""
        x = torch.zeros(1, 3, 8, 8)
        y = torch.ones(1, 3, 8, 8) * 0.5
        mse = compute_mse(x, y)
        assert mse == pytest.approx(0.25, rel=1e-5)

    def test_mse_returns_float(self):
        """MSE trả về float Python."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        assert isinstance(compute_mse(x, y), float)


class TestPSNR:
    """Test compute_psnr."""

    def test_identical_images_inf_psnr(self):
        """Ảnh giống nhau -> PSNR = inf."""
        x = torch.rand(2, 3, 64, 64)
        psnr = compute_psnr(x, x)
        assert psnr == float("inf"), f"PSNR phải = inf khi ảnh giống nhau: {psnr}"

    def test_psnr_nonnegative_for_small_noise(self):
        """PSNR >= 0 khi nhiễu nhỏ."""
        x = torch.rand(2, 3, 64, 64)
        noise = torch.randn_like(x) * 0.01
        y = (x + noise).clamp(0, 1)
        psnr = compute_psnr(x, y)
        assert psnr > 0, f"PSNR phải dương với nhiễu nhỏ: {psnr}"

    def test_psnr_decreases_with_more_noise(self):
        """Nhiễu tăng -> PSNR giảm."""
        torch.manual_seed(42)
        x = torch.rand(4, 3, 64, 64)
        noise_small = torch.randn_like(x) * 0.01
        noise_large = torch.randn_like(x) * 0.3
        psnr_high = compute_psnr(x, (x + noise_small).clamp(0, 1))
        psnr_low = compute_psnr(x, (x + noise_large).clamp(0, 1))
        assert psnr_high > psnr_low, "Nhiễu lớn hơn phải cho PSNR thấp hơn"

    def test_psnr_formula(self):
        """
        Kiểm tra công thức PSNR = 10 * log10(max_val^2 / MSE).
        """
        x = torch.zeros(1, 1, 4, 4)
        y = torch.ones(1, 1, 4, 4) * 0.1  # MSE = 0.01
        expected_psnr = 10 * np.log10(1.0 / 0.01)  # = 20 dB
        actual_psnr = compute_psnr(x, y, max_val=1.0)
        assert actual_psnr == pytest.approx(expected_psnr, rel=1e-5)

    def test_psnr_returns_float(self):
        """PSNR trả về float."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        assert isinstance(compute_psnr(x, y), float)


class TestSSIM:
    """Test compute_ssim."""

    def test_identical_images_max_ssim(self):
        """Ảnh giống nhau -> SSIM = 1.0."""
        x = torch.rand(2, 3, 64, 64)
        ssim = compute_ssim(x, x)
        assert ssim == pytest.approx(1.0, abs=1e-4), f"SSIM phải = 1 khi ảnh giống nhau: {ssim}"

    def test_ssim_in_range(self):
        """SSIM nằm trong khoảng [-1, 1]."""
        x = torch.rand(4, 3, 64, 64)
        y = torch.rand(4, 3, 64, 64)
        ssim = compute_ssim(x, y)
        assert -1 <= ssim <= 1, f"SSIM ngoài phạm vi: {ssim}"

    def test_ssim_decreases_with_noise(self):
        """Nhiễu tăng -> SSIM giảm."""
        torch.manual_seed(42)
        x = torch.rand(2, 3, 64, 64)
        noise_small = torch.randn_like(x) * 0.01
        noise_large = torch.randn_like(x) * 0.5
        ssim_high = compute_ssim(x, (x + noise_small).clamp(0, 1))
        ssim_low = compute_ssim(x, (x + noise_large).clamp(0, 1))
        assert ssim_high > ssim_low, "Nhiễu lớn hơn phải cho SSIM thấp hơn"

    def test_ssim_returns_float(self):
        """SSIM trả về float."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        assert isinstance(compute_ssim(x, y), float)

    def test_ssim_batch_consistency(self):
        """SSIM ổn định với batch size khác nhau."""
        torch.manual_seed(0)
        x1 = torch.rand(1, 3, 64, 64)
        x4 = x1.repeat(4, 1, 1, 1)
        y1 = torch.rand(1, 3, 64, 64)
        y4 = y1.repeat(4, 1, 1, 1)
        ssim1 = compute_ssim(x1, y1)
        ssim4 = compute_ssim(x4, y4)
        assert ssim1 == pytest.approx(ssim4, abs=1e-5)


class TestComputeAllMetrics:
    """Test compute_all_metrics trả về tuple (mse, psnr, ssim)."""

    def test_returns_tuple_of_three(self):
        """Trả về tuple 3 phần tử."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        result = compute_all_metrics(x, y)
        assert len(result) == 3

    def test_identical_images(self):
        """Ảnh giống nhau: MSE=0, PSNR=inf, SSIM≈1."""
        x = torch.rand(2, 3, 64, 64)
        mse, psnr, ssim = compute_all_metrics(x, x)
        assert mse == pytest.approx(0.0, abs=1e-7)
        assert psnr == float("inf")
        assert ssim == pytest.approx(1.0, abs=1e-4)

    def test_all_values_are_floats(self):
        """Tất cả output là float."""
        x = torch.rand(2, 3, 64, 64)
        y = torch.rand(2, 3, 64, 64)
        mse, psnr, ssim = compute_all_metrics(x, y)
        assert all(isinstance(v, float) for v in [mse, psnr, ssim])

    def test_consistency_with_individual_functions(self):
        """Kết quả từ compute_all_metrics khớp với từng hàm riêng lẻ."""
        torch.manual_seed(42)
        x = torch.rand(4, 3, 64, 64)
        y = torch.rand(4, 3, 64, 64)
        mse_all, psnr_all, ssim_all = compute_all_metrics(x, y)
        assert mse_all == pytest.approx(compute_mse(x, y), rel=1e-5)
        assert psnr_all == pytest.approx(compute_psnr(x, y), rel=1e-5)
        assert ssim_all == pytest.approx(compute_ssim(x, y), rel=1e-4)
