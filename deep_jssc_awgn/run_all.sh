#!/bin/bash
# =============================================================================
#  run_all.sh — Script chạy toàn bộ pipeline Deep JSCC AWGN từ C0 đến C8
# =============================================================================
# Sử dụng:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Script sẽ dừng lại nếu bất kỳ checkpoint nào FAIL.
# Cần GPU (hoặc CPU nếu chấp nhận thời gian lâu hơn).
# =============================================================================

set -e  # Dừng ngay khi có lỗi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[PASS]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[FAIL]${NC}  $1"; }

checkpoint_pass() {
    python tools/freeze_checkpoint.py --checkpoint "$1" --status PASS --note "Auto-run"
    log_ok "Checkpoint $1 đã đóng băng: PASS"
}

checkpoint_fail() {
    python tools/freeze_checkpoint.py --checkpoint "$1" --status FAIL --note "$2"
    log_error "Checkpoint $1 FAIL: $2"
    exit 1
}

echo ""
echo "========================================================"
echo "  Deep JSCC AWGN — Full Pipeline"
echo "  Bắt đầu: $(date)"
echo "========================================================"
echo ""

# =============================================================================
# C0: Kiểm tra môi trường
# =============================================================================
log_info "=== C0: Kiểm tra môi trường ==="
python -c "
import torch, torchvision, skimage, PIL, pandas, matplotlib, yaml, tqdm, pytest
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')
print('  Tất cả thư viện OK!')
" || checkpoint_fail "C0" "Import thư viện thất bại"
checkpoint_pass "C0"

# =============================================================================
# C1: Chuẩn bị dataset
# =============================================================================
log_info "=== C1: Chuẩn bị dataset CIFAR-10 ==="
python src/prepare_dataset.py --config configs/cbr_1_6.yaml \
    || checkpoint_fail "C1" "prepare_dataset.py thất bại"

python src/visualize_dataset.py --split train --num_images 16 \
    || log_warn "visualize_dataset không ảnh hưởng kết quả"
checkpoint_pass "C1"

# =============================================================================
# C2: Unit test AWGN + Metrics
# =============================================================================
log_info "=== C2: Unit tests AWGN + Metrics ==="
pytest tests/test_awgn.py tests/test_metrics.py -v \
    || checkpoint_fail "C2" "Unit test AWGN hoặc Metrics FAIL"
checkpoint_pass "C2"

# =============================================================================
# C3: Kiến trúc model
# =============================================================================
log_info "=== C3: Model architecture ==="
pytest tests/test_model_forward.py -v \
    || checkpoint_fail "C3" "Unit test model forward FAIL"

python src/print_model_summary.py --config configs/cbr_1_6.yaml \
    || checkpoint_fail "C3" "print_model_summary thất bại"
checkpoint_pass "C3"

# =============================================================================
# C4: Training CBR=1/6
# =============================================================================
log_info "=== C4: Training Deep JSCC CBR=1/6, SNR=10dB ==="
python src/train_jscc.py --config configs/cbr_1_6.yaml \
    || checkpoint_fail "C4" "Training CBR=1/6 thất bại"
checkpoint_pass "C4"

# =============================================================================
# C5: Monte Carlo evaluation CBR=1/6
# =============================================================================
log_info "=== C5: Monte Carlo eval CBR=1/6 ==="
python src/eval_mc.py \
    --config configs/cbr_1_6.yaml \
    --ckpt results/checkpoints/jscc_cbr_1_6_snr10_best.pt \
    || checkpoint_fail "C5" "eval_mc CBR=1/6 thất bại"
checkpoint_pass "C5"

# =============================================================================
# C6: Baseline evaluation
# =============================================================================
log_info "=== C6: Baseline JPEG+BPSK ==="
python src/eval_baseline.py --config configs/cbr_1_6.yaml \
    || checkpoint_fail "C6" "eval_baseline thất bại"
checkpoint_pass "C6"

# =============================================================================
# C7: Multi-CBR training và evaluation
# =============================================================================
log_info "=== C7: Training CBR=1/12 ==="
python src/train_jscc.py --config configs/cbr_1_12.yaml \
    || checkpoint_fail "C7" "Training CBR=1/12 thất bại"

log_info "=== C7: Training CBR=1/4 ==="
python src/train_jscc.py --config configs/cbr_1_4.yaml \
    || checkpoint_fail "C7" "Training CBR=1/4 thất bại"

log_info "=== C7: Monte Carlo eval CBR=1/12 ==="
python src/eval_mc.py \
    --config configs/cbr_1_12.yaml \
    --ckpt results/checkpoints/jscc_cbr_1_12_best.pt \
    || checkpoint_fail "C7" "eval_mc CBR=1/12 thất bại"

log_info "=== C7: Monte Carlo eval CBR=1/4 ==="
python src/eval_mc.py \
    --config configs/cbr_1_4.yaml \
    --ckpt results/checkpoints/jscc_cbr_1_4_best.pt \
    || checkpoint_fail "C7" "eval_mc CBR=1/4 thất bại"

checkpoint_pass "C7"

# =============================================================================
# C8: Sinh hình vẽ và bảng biểu
# =============================================================================
log_info "=== C8: Sinh figures và tables ==="
python src/plot_results.py \
    --input_dir results/tables \
    --output_dir results/figures \
    || checkpoint_fail "C8" "plot_results thất bại"
checkpoint_pass "C8"

# =============================================================================
# Hiển thị trạng thái cuối
# =============================================================================
echo ""
echo "========================================================"
echo "  ✅ Toàn bộ pipeline hoàn tất!"
echo "  Kết thúc: $(date)"
echo "========================================================"
python tools/freeze_checkpoint.py --show
echo ""
log_info "Hình vẽ: results/figures/"
log_info "Bảng:    results/tables/"
log_info "Model:   results/checkpoints/"
