"""
AWGN Channel Module
-------------------
Mô phỏng kênh truyền AWGN (Additive White Gaussian Noise).

Công thức: y = x + n, trong đó n ~ N(0, sigma^2)
với sigma^2 = signal_power / (2 * SNR_linear) cho tín hiệu phức
hoặc sigma^2 = signal_power / SNR_linear cho tín hiệu thực.
"""

import torch
import torch.nn as nn
import math


class AWGNChannel(nn.Module):
    """
    Kênh AWGN thực.

    Tín hiệu đầu vào x được chuẩn hoá năng lượng trước khi thêm nhiễu.
    Công suất nhiễu được tính từ SNR (dB) theo công thức:
        sigma^2 = 1 / (2 * 10^(snr_db/10))

    Tín hiệu đầu vào được chuẩn hoá về công suất trung bình = 1.
    """

    def __init__(self, snr_db: float = 10.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Truyền tín hiệu qua kênh AWGN.

        Args:
            x: Tensor tín hiệu đầu vào shape bất kỳ, dtype float32.

        Returns:
            Tensor cùng shape với x sau khi thêm nhiễu AWGN.
        """
        # Chuẩn hoá công suất tín hiệu về 1 (trên từng sample)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        power = x_flat.pow(2).mean(dim=1, keepdim=True)  # (B, 1)
        x_normalized = x_flat / (torch.sqrt(power) + 1e-8)

        # Tính sigma^2 từ SNR dB
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        noise_std = math.sqrt(1.0 / (2.0 * snr_linear))

        # Sinh nhiễu Gauss
        noise = torch.randn_like(x_normalized) * noise_std
        y = x_normalized + noise

        # Reshape về shape ban đầu
        return y.view_as(x)

    def set_snr(self, snr_db: float):
        """Cập nhật SNR (dB)."""
        self.snr_db = snr_db

    def get_noise_std(self) -> float:
        """Trả về độ lệch chuẩn nhiễu ứng với SNR hiện tại."""
        snr_linear = 10.0 ** (self.snr_db / 10.0)
        return math.sqrt(1.0 / (2.0 * snr_linear))

    def __repr__(self):
        return f"AWGNChannel(snr_db={self.snr_db:.1f} dB, noise_std={self.get_noise_std():.4f})"


def snr_db_to_sigma(snr_db: float) -> float:
    """Chuyển SNR (dB) sang độ lệch chuẩn nhiễu."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    return math.sqrt(1.0 / (2.0 * snr_linear))


def compute_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """
    Tính SNR thực tế (dB) từ signal và noise tensor.

    Args:
        signal: Tensor tín hiệu gốc.
        noise: Tensor nhiễu thêm vào.

    Returns:
        SNR tính bằng dB.
    """
    signal_power = signal.pow(2).mean().item()
    noise_power = noise.pow(2).mean().item()
    if noise_power == 0:
        return float("inf")
    snr_linear = signal_power / noise_power
    return 10.0 * math.log10(snr_linear)
