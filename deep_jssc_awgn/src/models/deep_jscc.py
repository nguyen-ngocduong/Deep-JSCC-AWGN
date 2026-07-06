"""
Deep JSCC Model
---------------
Kiến trúc Deep Joint Source-Channel Coding (Deep JSCC) cho ảnh qua kênh AWGN.

Tài liệu tham khảo:
    Bourtsoulatze et al., "Deep Joint Source-Channel Coding for Wireless Image Transmission",
    IEEE Transactions on Cognitive Communications and Networking, 2019.

Kiến trúc:
    Encoder: Conv layers -> bottleneck (channel symbols)
    Channel: AWGN
    Decoder: Transposed Conv layers -> reconstructed image
"""

import torch
import torch.nn as nn
from src.channels.awgn import AWGNChannel


class ConvBNReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ConvBNPReLU(nn.Sequential):
    """Conv2d + BatchNorm2d + PReLU (dùng trong encoder bottleneck)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        )


class DeepJSCCEncoder(nn.Module):
    """
    Encoder của Deep JSCC.

    Chuyển ảnh đầu vào (C, H, W) thành vector channel symbols có độ dài n_symbols.
    Kiến trúc: 4 Conv blocks downsampling + 1 Conv bottleneck.

    Args:
        img_channels: Số kênh màu ảnh đầu vào (3 cho RGB).
        n_symbols: Số channel symbols đầu ra (= CBR * C * H * W).
        base_channels: Số feature maps cơ sở.
    """

    def __init__(self, img_channels: int = 3, n_symbols: int = 2048,
                 base_channels: int = 256):
        super().__init__()
        # 4 bước downsampling: 64->32->16->8->4
        self.enc = nn.Sequential(
            ConvBNPReLU(img_channels, base_channels // 4, 9, stride=2, padding=4),  # 32x32
            ConvBNPReLU(base_channels // 4, base_channels // 2, 5, stride=2, padding=2),  # 16x16
            ConvBNPReLU(base_channels // 2, base_channels, 5, stride=2, padding=2),  # 8x8
            ConvBNPReLU(base_channels, base_channels, 5, stride=2, padding=2),  # 4x4
        )
        # Feature map size sau 4x downsampling: base_channels x 4 x 4 = base_channels*16
        self.flat_dim = base_channels * 4 * 4
        self.fc = nn.Linear(self.flat_dim, n_symbols)
        self.n_symbols = n_symbols

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Ảnh đầu vào [B, 3, 64, 64], giá trị trong [0, 1].
        Returns:
            Channel symbols [B, n_symbols].
        """
        feat = self.enc(x)
        feat_flat = feat.view(feat.size(0), -1)
        symbols = self.fc(feat_flat)
        return symbols


class DeepJSCCDecoder(nn.Module):
    """
    Decoder của Deep JSCC.

    Nhận channel symbols đầu ra từ kênh và khôi phục lại ảnh.

    Args:
        img_channels: Số kênh ảnh đầu ra (3 cho RGB).
        n_symbols: Số channel symbols đầu vào.
        base_channels: Số feature maps cơ sở.
        img_size: Kích thước ảnh đầu ra (mặc định 64).
    """

    def __init__(self, img_channels: int = 3, n_symbols: int = 2048,
                 base_channels: int = 256, img_size: int = 64):
        super().__init__()
        self.flat_dim = base_channels * 4 * 4
        self.base_channels = base_channels
        self.fc = nn.Linear(n_symbols, self.flat_dim)

        # 4 bước upsampling: 4->8->16->32->64
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(base_channels),
            nn.PReLU(),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.PReLU(),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.PReLU(),
            nn.ConvTranspose2d(base_channels // 4, img_channels, 9, stride=2, padding=4, output_padding=1),
            nn.Sigmoid(),  # Output trong [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Channel symbols sau kênh [B, n_symbols].
        Returns:
            Ảnh khôi phục [B, 3, 64, 64], giá trị trong [0, 1].
        """
        feat = self.fc(z)
        feat = feat.view(feat.size(0), self.base_channels, 4, 4)
        img = self.dec(feat)
        return img


class DeepJSCC(nn.Module):
    """
    Deep JSCC hoàn chỉnh: Encoder + AWGN Channel + Decoder.

    Args:
        img_channels: Số kênh màu (3 cho RGB).
        n_symbols: Số channel symbols (= CBR * C * H * W).
        base_channels: Số feature maps cơ sở.
        train_snr_db: SNR (dB) khi huấn luyện.
        img_size: Kích thước ảnh đầu vào/đầu ra.
    """

    def __init__(
        self,
        img_channels: int = 3,
        n_symbols: int = 2048,
        base_channels: int = 256,
        train_snr_db: float = 10.0,
        img_size: int = 64,
    ):
        super().__init__()
        self.encoder = DeepJSCCEncoder(img_channels, n_symbols, base_channels)
        self.channel = AWGNChannel(snr_db=train_snr_db)
        self.decoder = DeepJSCCDecoder(img_channels, n_symbols, base_channels, img_size)
        self.n_symbols = n_symbols
        self.img_channels = img_channels
        self.img_size = img_size

    def forward(self, x: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """
        Forward pass: encode -> channel -> decode.

        Args:
            x: Ảnh đầu vào [B, C, H, W], giá trị trong [0, 1].
            snr_db: SNR (dB) để dùng trong inference (None = dùng train SNR).

        Returns:
            Ảnh khôi phục [B, C, H, W], giá trị trong [0, 1].
        """
        if snr_db is not None:
            self.channel.set_snr(snr_db)

        symbols = self.encoder(x)          # [B, n_symbols]
        noisy_symbols = self.channel(symbols)  # [B, n_symbols]
        x_hat = self.decoder(noisy_symbols)    # [B, C, H, W]
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Chỉ chạy encoder, trả về channel symbols."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Chỉ chạy decoder từ channel symbols."""
        return self.decoder(z)

    def get_cbr(self) -> float:
        """Trả về Channel Bandwidth Ratio."""
        total_pixels = self.img_channels * self.img_size * self.img_size
        return self.n_symbols / total_pixels

    def count_parameters(self) -> int:
        """Đếm tổng số tham số trainable."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> DeepJSCC:
    """
    Tạo model DeepJSCC từ config dict.

    Args:
        config: Dict chứa các key: jscc.n_symbols, channel.train_snr_db, image_size.

    Returns:
        Đối tượng DeepJSCC.
    """
    jscc_cfg = config.get("jscc", {})
    channel_cfg = config.get("channel", {})

    n_symbols = jscc_cfg.get("n_symbols", 2048)
    train_snr_db = channel_cfg.get("train_snr_db", 10.0)
    img_size = config.get("image_size", 64)

    model = DeepJSCC(
        img_channels=3,
        n_symbols=n_symbols,
        base_channels=256,
        train_snr_db=float(train_snr_db),
        img_size=img_size,
    )
    return model
