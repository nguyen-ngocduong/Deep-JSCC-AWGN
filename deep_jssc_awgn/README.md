# Deep JSCC AWGN — Tài liệu mô tả toàn bộ dự án

> **Mục tiêu**: Triển khai mô phỏng Deep Joint Source-Channel Coding (Deep JSCC) cho truyền ảnh qua kênh AWGN, so sánh với baseline JPEG+BPSK theo phương pháp Monte Carlo.

---

## Mục lục

1. [Tổng quan lý thuyết](#1-tổng-quan-lý-thuyết)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Hệ thống Checkpoint C0–C9](#3-hệ-thống-checkpoint-c0c9)
4. [Hướng dẫn cài đặt](#4-hướng-dẫn-cài-đặt)
5. [Chạy từng bước](#5-chạy-từng-bước)
6. [Chạy toàn bộ pipeline](#6-chạy-toàn-bộ-pipeline)
7. [Kiến trúc mô hình](#7-kiến-trúc-mô-hình)
8. [Kênh AWGN và Monte Carlo](#8-kênh-awgn-và-monte-carlo)
9. [Baseline truyền thống](#9-baseline-truyền-thống)
10. [Kết quả bắt buộc](#10-kết-quả-bắt-buộc)
11. [Phân tích kết quả](#11-phân-tích-kết-quả)
12. [Checklist nộp bài](#12-checklist-nộp-bài)

---

## 1. Tổng quan lý thuyết

### 1.1 Deep JSCC là gì?

**Deep Joint Source-Channel Coding (Deep JSCC)** là phương pháp truyền thông tin tích hợp mã hoá nguồn và mã hoá kênh trong một mạng neural duy nhất, thay vì xử lý riêng lẻ như trong hệ thống truyền thống.

```
Hệ thống truyền thống:
  Ảnh → [Nén (JPEG)] → [Mã hóa kênh] → [Kênh AWGN] → [Giải mã kênh] → [Giải nén] → Ảnh

Hệ thống Deep JSCC:
  Ảnh → [Encoder CNN] → [Kênh AWGN] → [Decoder CNN] → Ảnh
```

**Tài liệu gốc**: Bourtsoulatze et al., *"Deep Joint Source-Channel Coding for Wireless Image Transmission"*, IEEE Trans. Cognitive Commun., 2019.

### 1.2 Channel Bandwidth Ratio (CBR)

$$\text{CBR} = \frac{k}{n} = \frac{\text{số channel symbols}}{\text{tổng số pixel (C × H × W)}}$$

| CBR | n_symbols | Ghi chú |
|-----|-----------|---------|
| 1/12 | 1024 | Băng thông hẹp nhất |
| 1/6  | 2048 | Thiết lập mặc định |
| 1/4  | 3072 | Băng thông rộng nhất |

### 1.3 Kênh AWGN

$$y = x + n, \quad n \sim \mathcal{N}(0, \sigma^2)$$

$$\sigma = \sqrt{\frac{1}{2 \cdot \text{SNR}_{\text{linear}}}}, \quad \text{SNR}_{\text{linear}} = 10^{\text{SNR}_{dB}/10}$$

Tín hiệu $x$ được **chuẩn hoá năng lượng** trước khi truyền: $\mathbb{E}[x^2] = 1$.

### 1.4 Metrics đánh giá

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| MSE | $\frac{1}{N}\sum(x_i - \hat{x}_i)^2$ | Lỗi bình phương trung bình |
| PSNR | $10\log_{10}(1/\text{MSE})$ (dB) | Tỷ lệ tín hiệu/nhiễu đỉnh |
| SSIM | Cấu trúc tương đồng | $\in [0,1]$, 1 là hoàn hảo |

---

## 2. Cấu trúc thư mục

```
deep_jssc_awgn/
│
├── configs/                    # Cấu hình YAML cho từng CBR
│   ├── cbr_1_12.yaml          # CBR = 1/12, n_symbols=1024
│   ├── cbr_1_6.yaml           # CBR = 1/6,  n_symbols=2048 (mặc định)
│   └── cbr_1_4.yaml           # CBR = 1/4,  n_symbols=3072
│
├── data/
│   ├── raw/                    # CIFAR-10 raw (tự tải về)
│   └── splits/                 # CSV splits
│       ├── train.csv           # 45,000 ảnh train
│       ├── val.csv             # 5,000 ảnh validation
│       └── test.csv            # 10,000 ảnh test
│
├── src/                        # Source code chính
│   ├── channels/
│   │   └── awgn.py            # AWGNChannel module
│   ├── metrics/
│   │   └── image_metrics.py   # MSE, PSNR, SSIM
│   ├── models/
│   │   └── deep_jscc.py       # Kiến trúc DeepJSCC
│   ├── baselines/
│   │   └── jpeg_bpsk.py       # Baseline re-exports
│   ├── data.py                 # DataLoader CIFAR-10
│   ├── prepare_dataset.py      # Tải và chia dataset
│   ├── visualize_dataset.py    # Visualize mẫu ảnh
│   ├── train_jscc.py           # Training loop
│   ├── eval_mc.py              # Monte Carlo evaluation
│   ├── eval_baseline.py        # Baseline evaluation
│   ├── plot_results.py         # Sinh hình/bảng báo cáo
│   └── print_model_summary.py  # In thông tin model
│
├── tests/                      # Unit tests (pytest)
│   ├── test_awgn.py            # Test AWGN channel
│   ├── test_metrics.py         # Test MSE/PSNR/SSIM
│   └── test_model_forward.py   # Test forward pass
│
├── results/
│   ├── checkpoints/            # Model checkpoints (.pt)
│   ├── figures/                # Hình vẽ (.png)
│   ├── tables/                 # Bảng kết quả (.csv)
│   └── logs/                   # Log file
│
├── frozen/                     # Manifest đóng băng C0-C9
│   ├── C0_environment/manifest.json
│   ├── C1_dataset/manifest.json
│   └── ...
│
├── report/                     # Báo cáo khoa học
│   ├── main.tex
│   └── final_report.pdf
│
├── tools/
│   └── freeze_checkpoint.py    # Tool đóng băng checkpoint
│
├── requirements.txt
├── run_all.sh                  # Script chạy toàn bộ pipeline
└── README.md
```

---

## 3. Hệ thống Checkpoint C0–C9

Dự án được chia thành **10 checkpoint** (C0-C9), mỗi checkpoint có mục tiêu, file bắt buộc, và acceptance criteria riêng.

```
C0: Kiểm tra môi trường
  └─► C1: Dataset preparation
        └─► C2: Unit tests AWGN + Metrics
              └─► C3: Model architecture
                    └─► C4: Training CBR=1/6
                          └─► C5: Monte Carlo eval CBR=1/6
                                └─► C6: Baseline eval
                                      └─► C7: Multi-CBR (1/12, 1/4)
                                            └─► C8: Figures & Tables
                                                  └─► C9: Report
```

> **Quy tắc**: Không chạy checkpoint tiếp theo nếu checkpoint trước còn FAIL.

| Checkpoint | Mô tả | Files sinh ra |
|-----------|-------|--------------|
| C0 | Kiểm tra import thư viện | — |
| C1 | Tải CIFAR-10, tạo CSV splits | `data/splits/*.csv`, `dataset_summary.csv` |
| C2 | pytest AWGN + Metrics | `results/tables/C2_unit_tests.csv` |
| C3 | pytest model + print summary | `model_summary_cbr_1_6.csv` |
| C4 | Train Deep JSCC CBR=1/6 | `jscc_cbr_1_6_snr10_best.pt`, `loss_curve_cbr_1_6.png` |
| C5 | Monte Carlo eval CBR=1/6 | `jscc_cbr_1_6_mc_raw.csv`, `jscc_cbr_1_6_mc_summary.csv` |
| C6 | Baseline JPEG+BPSK | `baseline_mc_raw.csv`, `baseline_mc_summary.csv` |
| C7 | Train + eval CBR=1/12, 1/4 | `jscc_cbr_1_12_best.pt`, `jscc_cbr_1_4_best.pt`, `jscc_all_cbr_mc_summary.csv` |
| C8 | Figures + Tables cuối | `final_psnr_vs_snr.png`, `final_ssim_vs_snr.png`, `final_psnr_ssim_table.csv` |
| C9 | Báo cáo PDF | `report/final_report.pdf` |

---

## 4. Hướng dẫn cài đặt

### 4.1 Yêu cầu hệ thống

| Thành phần | Tối thiểu | Khuyến nghị |
|-----------|-----------|------------|
| RAM | 8 GB | 16 GB |
| GPU | Không bắt buộc | NVIDIA ≥ 8GB VRAM |
| Python | 3.8+ | 3.10+ |
| CUDA | — | 11.7+ |

> **Lưu ý với CPU (RAM 8GB, không GPU)**: Training có thể mất **8–24 giờ** cho 100 epochs. Có thể giảm epochs xuống 20–30 để test nhanh.

### 4.2 Cài đặt môi trường

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-image Pillow pandas matplotlib tqdm pyyaml pytest

# Kiểm tra
python -c "import torch; print(torch.__version__)"
```

---

## 5. Chạy từng bước

### C0 — Kiểm tra môi trường

```bash
python -c "import torch, torchvision, skimage, PIL, pandas, matplotlib"
python tools/freeze_checkpoint.py --checkpoint C0 --status PASS
```

### C1 — Chuẩn bị dataset

```bash
# Tải CIFAR-10 và tạo splits (cần internet, ~170MB)
python src/prepare_dataset.py --config configs/cbr_1_6.yaml

# Kiểm tra visual
python src/visualize_dataset.py --split train --num_images 16

python tools/freeze_checkpoint.py --checkpoint C1 --status PASS
```

**File sinh ra:**
- `data/splits/train.csv` — 45,000 ảnh
- `data/splits/val.csv` — 5,000 ảnh
- `data/splits/test.csv` — 10,000 ảnh
- `results/tables/dataset_summary.csv`

### C2 — Unit tests AWGN + Metrics

```bash
pytest tests/test_awgn.py tests/test_metrics.py -v
python tools/freeze_checkpoint.py --checkpoint C2 --status PASS
```

**Expected:** Tất cả test PASS (≥ 20 test cases).

### C3 — Kiến trúc model

```bash
pytest tests/test_model_forward.py -v
python src/print_model_summary.py --config configs/cbr_1_6.yaml
python tools/freeze_checkpoint.py --checkpoint C3 --status PASS
```

**Expected output mẫu:**
```
CBR           : 0.166667 (1/6)
n_symbols     : 2048
Total params  : ~4.5M
Shape check PASSED
```

### C4 — Training Deep JSCC CBR=1/6

```bash
# Training 100 epochs (~2-24h tùy hardware)
python src/train_jscc.py --config configs/cbr_1_6.yaml

# Theo dõi log
tail -f results/logs/app.log
```

**Tham số training:**
| Tham số | Giá trị |
|---------|---------|
| Epochs | 100 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| LR scheduler | CosineAnnealing |
| Loss | MSE |
| Train SNR | 10 dB |
| Grad clip | norm=1.0 |

**File sinh ra:** `results/checkpoints/jscc_cbr_1_6_snr10_best.pt`

> **Lưu ý CPU**: Giảm epochs để test: sửa `train.epochs: 5` trong `cbr_1_6.yaml`

### C5 — Monte Carlo Evaluation CBR=1/6

```bash
python src/eval_mc.py \
    --config configs/cbr_1_6.yaml \
    --ckpt results/checkpoints/jscc_cbr_1_6_snr10_best.pt
```

**Cấu hình Monte Carlo:**
- K = 20 lần/SNR
- SNR test: 0, 5, 10, 15, 20 dB

**File sinh ra:**
- `results/tables/jscc_cbr_1_6_mc_raw.csv` — 100 dòng (5 SNR × 20 runs)
- `results/tables/jscc_cbr_1_6_mc_summary.csv` — 5 dòng (mean ± std)

### C6 — Baseline Evaluation

```bash
python src/eval_baseline.py --config configs/cbr_1_6.yaml
```

**Pipeline baseline:**
```
Ảnh → JPEG(Q) → bytes → bits → BPSK {±1} → AWGN → hard decision → JPEG decode → Ảnh
```

**File sinh ra:**
- `results/tables/baseline_mc_raw.csv`
- `results/tables/baseline_mc_summary.csv`
- `results/tables/baseline_bitrate_table.csv`

### C7 — Multi-CBR Training & Evaluation

```bash
# Train CBR=1/12
python src/train_jscc.py --config configs/cbr_1_12.yaml

# Train CBR=1/4
python src/train_jscc.py --config configs/cbr_1_4.yaml

# Eval Monte Carlo CBR=1/12
python src/eval_mc.py \
    --config configs/cbr_1_12.yaml \
    --ckpt results/checkpoints/jscc_cbr_1_12_best.pt

# Eval Monte Carlo CBR=1/4
python src/eval_mc.py \
    --config configs/cbr_1_4.yaml \
    --ckpt results/checkpoints/jscc_cbr_1_4_best.pt
```

### C8 — Sinh hình vẽ và bảng biểu

```bash
python src/plot_results.py \
    --input_dir results/tables \
    --output_dir results/figures
```

**Hình sinh ra:**

| File | Mô tả |
|------|-------|
| `final_psnr_vs_snr.png` | PSNR vs SNR với error bar (JSCC vs baseline) |
| `final_ssim_vs_snr.png` | SSIM vs SNR với error bar |
| `jscc_cbr_comparison_psnr.png` | So sánh CBR=1/12, 1/6, 1/4 |
| `reconstruction_grid.png` | Grid ảnh khôi phục theo SNR |
| `loss_curve_cbr_1_6.png` | Train/val loss curve |

---

## 6. Chạy toàn bộ pipeline

```bash
chmod +x run_all.sh
./run_all.sh
```

Script sẽ chạy tuần tự C0 → C8, dừng lại và báo lỗi nếu bất kỳ bước nào thất bại.

**Kiểm tra trạng thái checkpoints:**
```bash
python tools/freeze_checkpoint.py --show
```

---

## 7. Kiến trúc mô hình

### 7.1 Encoder

```
Input [B, 3, 64, 64]
  │
  ├── Conv(3→64, k=9, s=2, p=4) + BN + PReLU  → [B, 64, 32, 32]
  ├── Conv(64→128, k=5, s=2, p=2) + BN + PReLU → [B, 128, 16, 16]
  ├── Conv(128→256, k=5, s=2, p=2) + BN + PReLU → [B, 256, 8, 8]
  ├── Conv(256→256, k=5, s=2, p=2) + BN + PReLU → [B, 256, 4, 4]
  │   (Flatten: 256×4×4 = 4096)
  └── Linear(4096 → n_symbols)
      │
      Output symbols [B, n_symbols]
```

### 7.2 Kênh AWGN

```
symbols [B, n_symbols]
  │
  ├── Chuẩn hoá: x_norm = x / sqrt(E[x²])
  ├── Sinh nhiễu: n ~ N(0, σ²I), σ = sqrt(1/(2·SNR))
  └── y = x_norm + n
      │
      Output [B, n_symbols]
```

### 7.3 Decoder

```
Input [B, n_symbols]
  │
  ├── Linear(n_symbols → 4096)
  │   (Reshape: → [B, 256, 4, 4])
  ├── ConvT(256→256, k=5, s=2) + BN + PReLU  → [B, 256, 8, 8]
  ├── ConvT(256→128, k=5, s=2) + BN + PReLU  → [B, 128, 16, 16]
  ├── ConvT(128→64, k=5, s=2) + BN + PReLU   → [B, 64, 32, 32]
  └── ConvT(64→3, k=9, s=2) + Sigmoid         → [B, 3, 64, 64]
      │
      Output ảnh khôi phục [B, 3, 64, 64] ∈ [0,1]
```

### 7.4 Số tham số ước tính

| Phần | Tham số (ước tính) |
|------|-------------------|
| Encoder | ~4.2M |
| Decoder | ~4.5M |
| **Tổng** | **~8.7M** |

---

## 8. Kênh AWGN và Monte Carlo

### 8.1 Tại sao dùng Monte Carlo?

Kênh AWGN sinh ngẫu nhiên → mỗi lần inference cho kết quả khác nhau → cần chạy **K lần** rồi lấy mean ± std để có kết quả **thống kê ổn định**.

### 8.2 Quy trình Monte Carlo

```
Với mỗi SNR ∈ {0, 5, 10, 15, 20} dB:
  Với mỗi run k ∈ {1, ..., K=20}:
    Chạy toàn bộ test set qua model
    Tính MSE, PSNR, SSIM trung bình của run k
  Tính mean và std của K runs
→ Báo cáo: PSNR = mean ± std
```

### 8.3 Failure rate

Failure rate = tỷ lệ batch có PSNR < 5 dB. Quan trọng để báo cáo baseline JPEG khi SNR thấp (JPEG decode thất bại).

---

## 9. Baseline truyền thống

### 9.1 JPEG + BPSK

```
Ảnh → JPEG(Q) → bits → BPSK{±1} → +AWGN → hard decision → bits → JPEG decode → Ảnh
```

- **Vấn đề**: Ở SNR thấp, BER cao → bits lỗi → JPEG header hỏng → decode thất bại hoàn toàn → **cliff effect**

### 9.2 JPEG + Repetition Code + BPSK

```
bits → lặp 3 lần → [b,b,b,...] → BPSK → +AWGN → majority voting → bits → JPEG decode
```

- **Ưu điểm**: Giảm BER nhờ lặp bit
- **Nhược điểm**: Tốn băng thông gấp 3 lần (CBR tăng 3×)
- **Hạn chế**: Repetition code không tối ưu như LDPC/Polar code

### 9.3 CBR của baseline

$$\text{CBR}_{\text{baseline}} = \frac{\text{kích thước JPEG (bytes)} \times 8 \times \text{rep\_factor}}{C \times H \times W}$$

CBR của baseline phụ thuộc vào quality Q và nội dung ảnh, không cố định như Deep JSCC.

---

## 10. Kết quả bắt buộc

### 10.1 Bảng thông số mô phỏng

| Tham số | Giá trị |
|---------|---------|
| Dataset | CIFAR-10 |
| Image size | 64×64 pixels |
| Channel | AWGN |
| SNR test | 0, 5, 10, 15, 20 dB |
| Train SNR | 10 dB |
| Monte Carlo K | 20 lần/SNR |
| CBR | 1/12, 1/6, 1/4 |
| Loss | MSE |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 100 |
| Metrics | MSE, PSNR, SSIM |
| Baseline 1 | JPEG + BPSK |
| Baseline 2 | JPEG + Repetition(×3) + BPSK |

### 10.2 Cấu trúc bảng kết quả chính

| Model | CBR | SNR (dB) | PSNR mean±std | SSIM mean±std | MSE mean±std | Failure rate |
|-------|-----|----------|--------------|--------------|-------------|-------------|
| Deep JSCC | 1/6 | 0 | ... | ... | ... | 0 |
| Deep JSCC | 1/6 | 5 | ... | ... | ... | 0 |
| ... | ... | ... | ... | ... | ... | ... |
| JPEG+BPSK | ~1/6 | 5 | ... | ... | ... | ... |

### 10.3 Hình bắt buộc trong báo cáo

| Hình | Tên | File |
|------|-----|------|
| 1 | Sơ đồ Deep JSCC qua AWGN | Vẽ trong báo cáo |
| 2 | Sơ đồ checkpoint | Từ tài liệu hướng dẫn |
| 3 | Train/val loss curve | `results/figures/loss_curve_*.png` |
| 4 | PSNR vs SNR có error bar | `results/figures/final_psnr_vs_snr.png` |
| 5 | SSIM vs SNR có error bar | `results/figures/final_ssim_vs_snr.png` |
| 6 | Ảnh khôi phục theo SNR | `results/figures/reconstruction_grid.png` |
| 7 | Ảnh hưởng CBR đến PSNR | `results/figures/jscc_cbr_comparison_psnr.png` |

---

## 11. Phân tích kết quả

### 11.1 Xu hướng kỳ vọng

- **SNR tăng** → Nhiễu AWGN giảm → PSNR và SSIM tăng
- **CBR tăng** (1/12 → 1/6 → 1/4) → Decoder có nhiều thông tin hơn → PSNR/SSIM tăng
- **Deep JSCC**: Suy giảm **mềm (graceful degradation)** khi SNR giảm
- **JPEG+BPSK**: Suy giảm **đột ngột (cliff effect)** khi SNR xuống dưới ngưỡng

### 11.2 Câu kết luận mẫu

> Kết quả mô phỏng cho thấy mô hình Deep JSCC có thể tái tạo ảnh qua kênh AWGN với chất lượng tăng dần theo SNR và CBR. So với baseline JPEG+BPSK và JPEG+Repetition+BPSK trong cùng thiết lập, Deep JSCC có xu hướng suy giảm mềm hơn ở vùng SNR thấp. Việc đánh giá Monte Carlo (K=20) cho phép báo cáo kết quả ổn định hơn thông qua giá trị trung bình và độ lệch chuẩn của PSNR, SSIM và MSE.

### 11.3 Lưu ý khi viết báo cáo

- ❌ **Không kết luận** Deep JSCC luôn tốt hơn mọi hệ thống nếu chỉ so với repetition code
- ✅ **Nên viết**: "Trong thiết lập mô phỏng này, Deep JSCC cho xu hướng ổn định hơn ở SNR thấp"
- ✅ **Ghi rõ hạn chế**: Baseline mới chỉ là JPEG+Repetition, chưa thử LDPC/Polar code
- ✅ **Báo cáo failure rate** thay vì bỏ qua các mẫu JPEG decode thất bại

---

## 12. Checklist nộp bài

### Code & Scripts

- [ ] Repo có đầy đủ `src/`, `configs/`, `tests/`, `results/` (sample)
- [ ] `run_all.sh` chạy được từ đầu đến cuối
- [ ] `README.md` có lệnh chạy lại từng checkpoint

### Dataset

- [ ] `data/splits/train.csv`, `val.csv`, `test.csv`
- [ ] `results/tables/dataset_summary.csv`

### Model

- [ ] Best checkpoint `jscc_cbr_1_6_snr10_best.pt`
- [ ] Checkpoint cho CBR=1/12 và CBR=1/4

### Monte Carlo Results

- [ ] Raw CSV và summary CSV cho JSCC (3 CBR × 5 SNR)
- [ ] Raw CSV và summary CSV cho baseline

### Figures

- [ ] `final_psnr_vs_snr.png` (với error bar)
- [ ] `final_ssim_vs_snr.png` (với error bar)
- [ ] `reconstruction_grid.png`
- [ ] `jscc_cbr_comparison_psnr.png`
- [ ] `loss_curve_cbr_1_6.png`

### Report

- [ ] PDF báo cáo khoa học
- [ ] Mọi số liệu trong bài khớp với CSV ở C8
- [ ] Mọi hình có caption và được nhắc trong nội dung
- [ ] Có mục phân tích hạn chế của baseline
- [ ] Có mô tả Monte Carlo K lần

### Reproducibility

- [ ] `frozen/*/manifest.json` với status PASS cho C0-C8
- [ ] `requirements.txt` đầy đủ

---

## Tham khảo

1. Bourtsoulatze, E., Kurka, D. B., & Gündüz, D. (2019). *Deep joint source-channel coding for wireless image transmission*. IEEE Transactions on Cognitive Communications and Networking, 5(3), 567-579.
2. Cover, T. M., & Thomas, J. A. (2006). *Elements of information theory*. John Wiley & Sons.
3. Shannon, C. E. (1948). *A mathematical theory of communication*. Bell System Technical Journal, 27(3), 379-423.
