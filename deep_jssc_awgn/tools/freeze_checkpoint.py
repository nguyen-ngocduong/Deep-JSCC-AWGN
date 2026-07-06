"""
Freeze Checkpoint Tool
-----------------------
Đóng băng kết quả một checkpoint (C0-C9) bằng cách cập nhật manifest.json
và lưu metadata mô tả trạng thái hiện tại.

Sử dụng:
    python tools/freeze_checkpoint.py --checkpoint C4 --status PASS
    python tools/freeze_checkpoint.py --checkpoint C4 --status FAIL --note "Training chưa hội tụ"
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


CHECKPOINTS = {
    "C0": {
        "name": "C0_environment",
        "description": "Kiểm tra môi trường (torch, torchvision, skimage, PIL, pandas, matplotlib)",
        "required_files": [],
    },
    "C1": {
        "name": "C1_dataset",
        "description": "Chuẩn bị và chia dataset CIFAR-10",
        "required_files": [
            "data/splits/train.csv",
            "data/splits/val.csv",
            "data/splits/test.csv",
            "results/tables/dataset_summary.csv",
        ],
    },
    "C2": {
        "name": "C2_awgn_metrics",
        "description": "Unit test AWGN channel và image metrics",
        "required_files": [
            "src/channels/awgn.py",
            "src/metrics/image_metrics.py",
            "tests/test_awgn.py",
            "tests/test_metrics.py",
        ],
    },
    "C3": {
        "name": "C3_model_architecture",
        "description": "Kiến trúc mô hình Deep JSCC",
        "required_files": [
            "src/models/deep_jscc.py",
            "tests/test_model_forward.py",
            "results/tables/model_summary_cbr_1_6.csv",
        ],
    },
    "C4": {
        "name": "C4_train_single_config",
        "description": "Training Deep JSCC với CBR=1/6, SNR=10dB",
        "required_files": [
            "results/checkpoints/jscc_cbr_1_6_snr10_best.pt",
            "results/tables/train_jscc_cbr_1_6_snr10.csv",
            "results/figures/loss_curve_cbr_1_6.png",
        ],
    },
    "C5": {
        "name": "C5_monte_carlo_eval",
        "description": "Monte Carlo evaluation CBR=1/6",
        "required_files": [
            "results/tables/jscc_cbr_1_6_mc_raw.csv",
            "results/tables/jscc_cbr_1_6_mc_summary.csv",
        ],
    },
    "C6": {
        "name": "C6_baseline",
        "description": "Baseline JPEG+BPSK và JPEG+Repetition+BPSK",
        "required_files": [
            "results/tables/baseline_mc_raw.csv",
            "results/tables/baseline_mc_summary.csv",
            "results/tables/baseline_bitrate_table.csv",
        ],
    },
    "C7": {
        "name": "C7_multi_cbr",
        "description": "Training và eval cho CBR=1/12 và CBR=1/4",
        "required_files": [
            "results/checkpoints/jscc_cbr_1_12_best.pt",
            "results/checkpoints/jscc_cbr_1_4_best.pt",
            "results/tables/jscc_cbr_1_12_mc_summary.csv",
            "results/tables/jscc_all_cbr_mc_summary.csv",
            "results/figures/jscc_cbr_comparison_psnr.png",
        ],
    },
    "C8": {
        "name": "C8_results_figures",
        "description": "Tổng hợp hình vẽ và bảng biểu",
        "required_files": [
            "results/tables/final_psnr_ssim_table.csv",
            "results/tables/final_simulation_settings.csv",
            "results/figures/final_psnr_vs_snr.png",
            "results/figures/final_ssim_vs_snr.png",
            "results/figures/reconstruction_grid.png",
        ],
    },
    "C9": {
        "name": "C9_report",
        "description": "Báo cáo khoa học",
        "required_files": [
            "report/main.tex",
            "report/final_report.pdf",
        ],
    },
}


def check_required_files(checkpoint_id: str, base_dir: str = ".") -> dict:
    """Kiểm tra các file bắt buộc của checkpoint."""
    cp = CHECKPOINTS.get(checkpoint_id, {})
    required = cp.get("required_files", [])
    file_status = {}
    for f in required:
        full_path = os.path.join(base_dir, f)
        exists = os.path.isfile(full_path)
        size = os.path.getsize(full_path) if exists else 0
        file_status[f] = {
            "exists": exists,
            "size_bytes": size,
        }
    return file_status


def freeze_checkpoint(
    checkpoint_id: str,
    status: str,
    note: str = "",
    base_dir: str = ".",
):
    """
    Cập nhật manifest.json của checkpoint với trạng thái hiện tại.

    Args:
        checkpoint_id: Ví dụ 'C4'.
        status: 'PASS' hoặc 'FAIL'.
        note: Ghi chú thêm.
        base_dir: Thư mục gốc project.
    """
    cp = CHECKPOINTS.get(checkpoint_id)
    if cp is None:
        print(f"❌ Checkpoint không tồn tại: {checkpoint_id}")
        print(f"   Các checkpoint hợp lệ: {list(CHECKPOINTS.keys())}")
        sys.exit(1)

    manifest_dir = os.path.join(base_dir, "frozen", cp["name"])
    manifest_path = os.path.join(manifest_dir, "manifest.json")

    # Kiểm tra file bắt buộc
    file_status = check_required_files(checkpoint_id, base_dir)
    n_missing = sum(1 for v in file_status.values() if not v["exists"])

    if status == "PASS" and n_missing > 0:
        print(f"⚠️  Cảnh báo: {n_missing} file bắt buộc chưa tồn tại:")
        for f, info in file_status.items():
            if not info["exists"]:
                print(f"   ❌ {f}")

    # Tạo manifest
    manifest = {
        "checkpoint": checkpoint_id,
        "name": cp["name"],
        "description": cp["description"],
        "status": status,
        "frozen_at": datetime.now().isoformat(),
        "note": note,
        "required_files": file_status,
        "n_missing_files": n_missing,
    }

    # Đọc manifest cũ nếu có (giữ history)
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                old_manifest = json.loads(content)
                # Giữ history
                history = old_manifest.get("history", [])
                history.append({
                    "status": old_manifest.get("status"),
                    "frozen_at": old_manifest.get("frozen_at"),
                    "note": old_manifest.get("note"),
                })
                manifest["history"] = history
            else:
                # File rỗng — bỏ qua, không có history
                pass
        except json.JSONDecodeError as e:
            print(f"⚠️  manifest.json bị hỏng hoặc rỗng ({e}), ghi đè mới.")
            manifest["history"] = []

    os.makedirs(manifest_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    status_icon = "✅" if status == "PASS" else "❌"
    print(f"\n{status_icon} Checkpoint {checkpoint_id} ({cp['name']}) -> {status}")
    print(f"   Manifest: {manifest_path}")
    if note:
        print(f"   Ghi chú: {note}")
    if n_missing > 0:
        print(f"   ⚠️  {n_missing} file chưa có")
    else:
        print(f"   ✅ Tất cả {len(file_status)} file bắt buộc đều có")


def show_status(base_dir: str = "."):
    """Hiển thị trạng thái tất cả checkpoints."""
    print("\n" + "=" * 60)
    print("  Trạng thái các Checkpoint")
    print("=" * 60)
    for cp_id, cp_info in CHECKPOINTS.items():
        manifest_path = os.path.join(
            base_dir, "frozen", cp_info["name"], "manifest.json"
        )
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    manifest = json.loads(content)
                    status = manifest.get("status", "UNKNOWN")
                    frozen_at = manifest.get("frozen_at", "N/A")
                    icon = "✅" if status == "PASS" else "❌"
                    print(f"  {icon} {cp_id}: {status} (frozen at {frozen_at[:10]})")
                else:
                    print(f"  ⬜ {cp_id}: manifest.json rỗng")
            except json.JSONDecodeError:
                print(f"  ⚠️  {cp_id}: manifest.json bị hỏng")
        else:
            print(f"  ⬜ {cp_id}: chưa đóng băng")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Đóng băng checkpoint trong project Deep JSCC"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help="Checkpoint ID (C0-C9)",
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["PASS", "FAIL"],
        default="PASS",
        help="Trạng thái checkpoint",
    )
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Ghi chú thêm",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị trạng thái tất cả checkpoints",
    )
    args = parser.parse_args()

    if args.show:
        show_status()
        return

    if not args.checkpoint:
        parser.print_help()
        sys.exit(1)

    freeze_checkpoint(args.checkpoint, args.status, args.note)


if __name__ == "__main__":
    main()
