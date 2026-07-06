"""
Baseline: JPEG + BPSK module
-----------------------------
Triển khai baseline truyền thống để so sánh với Deep JSCC.
"""

# Module này chứa các hàm được dùng bởi src/eval_baseline.py
# Đọc eval_baseline.py để hiểu pipeline đầy đủ.

from src.eval_baseline import (
    jpeg_encode,
    jpeg_decode_safe,
    bytes_to_bits,
    bits_to_bpsk,
    add_awgn,
    bpsk_hard_decision,
    bits_to_bytes,
    apply_repetition_code,
    decode_repetition_code,
    compute_bitrate,
    tensor_to_pil,
    pil_to_tensor,
)

__all__ = [
    "jpeg_encode",
    "jpeg_decode_safe",
    "bytes_to_bits",
    "bits_to_bpsk",
    "add_awgn",
    "bpsk_hard_decision",
    "bits_to_bytes",
    "apply_repetition_code",
    "decode_repetition_code",
    "compute_bitrate",
    "tensor_to_pil",
    "pil_to_tensor",
]
