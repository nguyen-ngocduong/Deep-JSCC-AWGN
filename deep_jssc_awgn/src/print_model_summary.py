"""
Print Model Summary Script
---------------------------
In thông tin chi tiết về kiến trúc mô hình Deep JSCC:
- Số tham số từng layer
- Kích thước feature maps
- Tổng số tham số
- CBR

Sử dụng:
    python src/print_model_summary.py --config configs/cbr_1_6.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.deep_jscc import build_model


def count_parameters_per_layer(model: torch.nn.Module) -> pd.DataFrame:
    """Đếm tham số từng layer và trả về DataFrame."""
    rows = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            n_params = sum(p.numel() for p in module.parameters())
            n_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if n_params > 0:
                rows.append({
                    "layer": name,
                    "type": module.__class__.__name__,
                    "params": n_params,
                    "trainable": n_trainable,
                })
    return pd.DataFrame(rows)


def print_model_summary(config: dict):
    """In model summary chi tiết."""
    model = build_model(config)

    print("\n" + "=" * 70)
    print("  Deep JSCC Model Summary")
    print("=" * 70)

    # Thông tin tổng quát
    print(f"\n📊 Thông tin tổng quát:")
    print(f"   CBR           : {model.get_cbr():.6f} ({config.get('jscc', {}).get('cbr', 'N/A')})")
    print(f"   n_symbols     : {model.n_symbols}")
    print(f"   Image size    : {model.img_size}×{model.img_size}×{model.img_channels}")
    print(f"   Train SNR     : {config.get('channel', {}).get('train_snr_db', 10.0)} dB")
    print(f"   Total params  : {model.count_parameters():,}")
    print(f"   Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"   Decoder params: {sum(p.numel() for p in model.decoder.parameters()):,}")

    # Layer-wise summary
    print(f"\n📋 Chi tiết từng layer:")
    df = count_parameters_per_layer(model)
    print(df.to_string(index=False))

    # Kiểm tra forward pass
    print(f"\n🔎 Kiểm tra kích thước tensor (forward pass):")
    dummy_input = torch.zeros(1, 3, model.img_size, model.img_size)
    with torch.no_grad():
        symbols = model.encoder(dummy_input)
        recon = model(dummy_input)
    print(f"   Input shape   : {list(dummy_input.shape)}")
    print(f"   Symbols shape : {list(symbols.shape)}")
    print(f"   Output shape  : {list(recon.shape)}")
    assert recon.shape == dummy_input.shape, "❌ Input/Output shape không khớp!"
    print(f"   ✅ Shape check PASSED")

    # Lưu model summary CSV
    summary_path = f"results/tables/model_summary_{config.get('jscc', {}).get('cbr', 'unknown')}.csv"
    summary_path = summary_path.replace("/", "_").replace(".", "_")
    summary_path = f"results/tables/model_summary_cbr_1_6.csv"

    import os
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv(summary_path, index=False)
    print(f"\n💾 Model summary CSV lưu tại: {summary_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="In model summary của Deep JSCC")
    parser.add_argument("--config", type=str, default="configs/cbr_1_6.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print_model_summary(config)


if __name__ == "__main__":
    main()
