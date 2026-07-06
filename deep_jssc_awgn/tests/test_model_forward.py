"""
Unit Tests cho Model Forward Pass
-----------------------------------
Test: DeepJSCC model hoạt động đúng về shape và giá trị output.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.deep_jscc import DeepJSCC, build_model, DeepJSCCEncoder, DeepJSCCDecoder


# Config mẫu cho test (nhẹ để chạy nhanh)
TEST_CONFIG = {
    "jscc": {
        "cbr": 1 / 6,
        "n_symbols": 2048,
    },
    "channel": {
        "train_snr_db": 10.0,
        "test_snr_db": [0, 5, 10, 15, 20],
    },
    "image_size": 64,
    "seed": 42,
}

# Config nhẹ cho test nhanh
LIGHT_CONFIG = {
    "jscc": {
        "cbr": 1 / 6,
        "n_symbols": 192,  # 8x8x3 = minimal
    },
    "channel": {
        "train_snr_db": 10.0,
    },
    "image_size": 64,
}


class TestDeepJSCCForward:
    """Test forward pass của DeepJSCC."""

    @pytest.fixture
    def model(self):
        """Tạo model nhỏ cho testing."""
        return DeepJSCC(
            img_channels=3,
            n_symbols=2048,
            base_channels=64,  # Nhỏ hơn để test nhanh
            train_snr_db=10.0,
            img_size=64,
        )

    def test_output_shape_standard(self, model):
        """Output shape phải là [B, 3, 64, 64]."""
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape, f"Output shape {y.shape} != Input shape {x.shape}"

    def test_output_range(self, model):
        """Output phải nằm trong [0, 1] (vì Sigmoid cuối decoder)."""
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.min() >= 0.0, f"Output min < 0: {y.min().item()}"
        assert y.max() <= 1.0, f"Output max > 1: {y.max().item()}"

    def test_no_nan_output(self, model):
        """Output không chứa NaN."""
        x = torch.rand(4, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert not torch.isnan(y).any(), "Output chứa NaN!"

    def test_no_inf_output(self, model):
        """Output không chứa Inf."""
        x = torch.rand(4, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert not torch.isinf(y).any(), "Output chứa Inf!"

    def test_batch_size_1(self, model):
        """Hoạt động đúng với batch_size=1."""
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, 64, 64)

    def test_batch_size_8(self, model):
        """Hoạt động đúng với batch_size=8."""
        x = torch.rand(8, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (8, 3, 64, 64)

    def test_snr_override(self, model):
        """Có thể override SNR khi forward."""
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            y_snr5 = model(x, snr_db=5.0)
            y_snr20 = model(x, snr_db=20.0)
        assert y_snr5.shape == (2, 3, 64, 64)
        assert y_snr20.shape == (2, 3, 64, 64)

    def test_different_snr_different_output(self, model):
        """SNR khác nhau (với noise) cho output khác nhau."""
        torch.manual_seed(0)
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            y_low = model(x, snr_db=0.0)
            torch.manual_seed(1)
            y_high = model(x, snr_db=40.0)
        assert not torch.allclose(y_low, y_high, atol=1e-3), \
            "SNR khác nhau nên cho output khác nhau"

    def test_cbr_correct(self, model):
        """CBR tính đúng."""
        expected_cbr = 2048 / (3 * 64 * 64)
        actual_cbr = model.get_cbr()
        assert actual_cbr == pytest.approx(expected_cbr, rel=1e-5)

    def test_encoder_output_shape(self, model):
        """Encoder output shape = [B, n_symbols]."""
        x = torch.rand(4, 3, 64, 64)
        with torch.no_grad():
            symbols = model.encode(x)
        assert symbols.shape == (4, 2048), f"Encoder output shape sai: {symbols.shape}"

    def test_decoder_output_shape(self, model):
        """Decoder output shape = [B, 3, 64, 64]."""
        z = torch.rand(4, 2048)
        with torch.no_grad():
            recon = model.decode(z)
        assert recon.shape == (4, 3, 64, 64)

    def test_count_parameters_positive(self, model):
        """Số tham số phải > 0."""
        assert model.count_parameters() > 0

    def test_gradients_flow(self, model):
        """Gradient phải chạy qua được (no dead layers)."""
        model.train()
        x = torch.rand(2, 3, 64, 64)
        recon = model(x)
        loss = ((recon - x) ** 2).mean()
        loss.backward()
        # Kiểm tra ít nhất một param có gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "Gradient không chạy qua model!"


class TestBuildModel:
    """Test build_model từ config."""

    def test_cbr_1_6(self):
        """Build model với CBR=1/6."""
        config = {
            "jscc": {"cbr": 1 / 6, "n_symbols": 2048},
            "channel": {"train_snr_db": 10.0},
            "image_size": 64,
        }
        model = build_model(config)
        assert model is not None
        assert model.n_symbols == 2048

    def test_cbr_1_12(self):
        """Build model với CBR=1/12."""
        config = {
            "jscc": {"cbr": 1 / 12, "n_symbols": 1024},
            "channel": {"train_snr_db": 10.0},
            "image_size": 64,
        }
        model = build_model(config)
        assert model.n_symbols == 1024

    def test_cbr_1_4(self):
        """Build model với CBR=1/4."""
        config = {
            "jscc": {"cbr": 1 / 4, "n_symbols": 3072},
            "channel": {"train_snr_db": 10.0},
            "image_size": 64,
        }
        model = build_model(config)
        assert model.n_symbols == 3072

    def test_model_is_deep_jscc_instance(self):
        """build_model trả về instance DeepJSCC."""
        model = build_model(TEST_CONFIG)
        assert isinstance(model, DeepJSCC)


class TestDeepJSCCEncoder:
    """Test encoder riêng."""

    def test_encoder_output_shape(self):
        encoder = DeepJSCCEncoder(img_channels=3, n_symbols=512, base_channels=64)
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 512)

    def test_encoder_no_nan(self):
        encoder = DeepJSCCEncoder(img_channels=3, n_symbols=512, base_channels=64)
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            out = encoder(x)
        assert not torch.isnan(out).any()


class TestDeepJSCCDecoder:
    """Test decoder riêng."""

    def test_decoder_output_shape(self):
        decoder = DeepJSCCDecoder(img_channels=3, n_symbols=512, base_channels=64, img_size=64)
        z = torch.rand(2, 512)
        with torch.no_grad():
            out = decoder(z)
        assert out.shape == (2, 3, 64, 64)

    def test_decoder_output_range(self):
        decoder = DeepJSCCDecoder(img_channels=3, n_symbols=512, base_channels=64, img_size=64)
        z = torch.rand(2, 512)
        with torch.no_grad():
            out = decoder(z)
        assert out.min() >= 0.0
        assert out.max() <= 1.0
