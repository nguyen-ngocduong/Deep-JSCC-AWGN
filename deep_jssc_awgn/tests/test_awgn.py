"""
Unit Tests cho AWGN Channel
----------------------------
Test: AWGNChannel module hoạt động đúng theo spec.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.channels.awgn import AWGNChannel, snr_db_to_sigma, compute_snr_db


class TestAWGNChannel:
    """Test suite cho AWGNChannel."""

    def test_output_shape_preserved(self):
        """Output phải cùng shape với input."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(4, 128)
        y = channel(x)
        assert y.shape == x.shape, f"Shape mismatch: {y.shape} != {x.shape}"

    def test_output_shape_2d(self):
        """Test với tensor 2D."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(8, 512)
        y = channel(x)
        assert y.shape == x.shape

    def test_output_shape_4d(self):
        """Test với tensor 4D (image-like)."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(2, 3, 64, 64)
        y = channel(x)
        assert y.shape == x.shape

    def test_high_snr_low_noise(self):
        """Ở SNR cao, tín hiệu sau kênh gần giống tín hiệu gốc."""
        channel = AWGNChannel(snr_db=40.0)
        torch.manual_seed(42)
        x = torch.randn(16, 256)
        y = channel(x)
        # Normalize x như channel làm
        x_flat = x.view(x.shape[0], -1)
        power = x_flat.pow(2).mean(dim=1, keepdim=True)
        x_norm = x_flat / (torch.sqrt(power) + 1e-8)
        mse = ((x_norm - y.view(x.shape[0], -1)) ** 2).mean().item()
        # Ở SNR=40dB, MSE nhiễu ~ 1/(2*10^4) ≈ 5e-5
        assert mse < 0.01, f"MSE quá cao ở SNR=40dB: {mse:.6f}"

    def test_low_snr_high_noise(self):
        """Ở SNR thấp, MSE nhiễu phải lớn hơn SNR cao."""
        channel_high = AWGNChannel(snr_db=30.0)
        channel_low = AWGNChannel(snr_db=0.0)
        torch.manual_seed(42)
        x = torch.randn(16, 256)

        y_high = channel_high(x)
        y_low = channel_low(x)

        # Normalize x
        x_flat = x.view(x.shape[0], -1)
        power = x_flat.pow(2).mean(dim=1, keepdim=True)
        x_norm = x_flat / (torch.sqrt(power) + 1e-8)

        mse_high = ((x_norm - y_high.view(x.shape[0], -1)) ** 2).mean().item()
        mse_low = ((x_norm - y_low.view(x.shape[0], -1)) ** 2).mean().item()

        assert mse_low > mse_high, "SNR thấp phải cho MSE lớn hơn SNR cao"

    def test_noise_std_formula(self):
        """Kiểm tra công thức noise_std = sqrt(1/(2*SNR_linear))."""
        for snr_db in [0, 5, 10, 20]:
            sigma_expected = math.sqrt(1.0 / (2.0 * 10 ** (snr_db / 10)))
            sigma_actual = snr_db_to_sigma(snr_db)
            assert abs(sigma_expected - sigma_actual) < 1e-9, \
                f"SNR={snr_db} dB: expected={sigma_expected:.6f}, got={sigma_actual:.6f}"

    def test_set_snr(self):
        """Test set_snr thay đổi SNR đúng."""
        channel = AWGNChannel(snr_db=10.0)
        channel.set_snr(20.0)
        assert channel.snr_db == 20.0

    def test_no_nan_output(self):
        """Output không được chứa NaN."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(8, 256)
        y = channel(x)
        assert not torch.isnan(y).any(), "Output chứa NaN!"

    def test_no_inf_output(self):
        """Output không được chứa Inf."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(8, 256)
        y = channel(x)
        assert not torch.isinf(y).any(), "Output chứa Inf!"

    def test_deterministic_with_seed(self):
        """Cùng seed phải cho cùng output."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(4, 64)
        torch.manual_seed(123)
        y1 = channel(x)
        torch.manual_seed(123)
        y2 = channel(x)
        assert torch.allclose(y1, y2), "Cùng seed phải cho output giống nhau"

    def test_different_seeds_different_outputs(self):
        """Seed khác nhau phải cho output khác nhau (với xác suất cao)."""
        channel = AWGNChannel(snr_db=10.0)
        x = torch.randn(4, 64)
        torch.manual_seed(1)
        y1 = channel(x)
        torch.manual_seed(2)
        y2 = channel(x)
        assert not torch.allclose(y1, y2), "Seed khác nhau nên cho output khác nhau"


class TestAWGNUtilities:
    """Test các utility functions."""

    def test_snr_db_to_sigma_zero_db(self):
        """SNR = 0 dB -> sigma = sqrt(0.5) ≈ 0.7071."""
        sigma = snr_db_to_sigma(0.0)
        assert abs(sigma - math.sqrt(0.5)) < 1e-9

    def test_snr_db_to_sigma_ten_db(self):
        """SNR = 10 dB -> sigma = sqrt(1/20) ≈ 0.2236."""
        sigma = snr_db_to_sigma(10.0)
        expected = math.sqrt(1.0 / 20.0)
        assert abs(sigma - expected) < 1e-9

    def test_compute_snr_db_returns_float(self):
        """compute_snr_db trả về float."""
        signal = torch.randn(4, 64)
        noise = torch.randn(4, 64) * 0.1
        snr = compute_snr_db(signal, noise)
        assert isinstance(snr, float)

    def test_compute_snr_db_zero_noise(self):
        """Nhiễu = 0 -> SNR = inf."""
        signal = torch.randn(4, 64)
        noise = torch.zeros(4, 64)
        snr = compute_snr_db(signal, noise)
        assert snr == float("inf")
