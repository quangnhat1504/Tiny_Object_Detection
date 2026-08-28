---
title: Phase 2 Metric Chain Ablation - 2026-07-01
type: analysis
created: 2026-07-01
updated: 2026-07-01
sources: [raw/plan.md, common/model.py, common/metrics/igwd.py, common/metrics/__init__.py, common/eval_utils.py, common/config.py, scripts/train_frcnn_metric.py, scripts/run_phase2.py, bug.txt, raw/RFLA.pdf]
tags: [tiny-object-detection, phase-2, metric-ablation, sa-alw]
---

## Phase 2 Metric Chain Ablation - 2026-07-01

## Question

Cô lập đóng góp của từng component trong chuỗi metric: IoU → NWD → IGWD → IGWD-ablations → ALW → SA-ALW. Mỗi metric đóng góp bao nhiêu vào mAP?

## Phase 1 Recap

| Model | mAP@50 (mean±std) | mAP@50:95 |
|-------|:---:|:---:|
| YOLOv8n | 0.1705 | 0.0505 |
| YOLO11n | 0.1690 | 0.0520 |
| FRCNN full (IoU) | **0.3892 ± 0.0019** | 0.5822 |
| FRCNN patches | 0.3804 ± 0.0020 | 0.6075 |

**Key finding**: FRCNN mAP ~2.3× YOLO. Patches không cải thiện mAP@50 (thậm chí thấp hơn) nhưng tăng mAP@50:95 (0.582 → 0.607). Overfit sớm (best epoch 5-9/20) → cần early stopping cho Phase 2.

## Phase 2: 7 Configs

Tất cả chạy FRCNN full-image, `placement=la_loss` (metric trong RPN label assignment, box regression giữ Smooth-L1).

| Step | Metric key | Mô tả |
|:----:|------------|-------|
| 2.1 | `ciou` (IoU) | = Phase 1 baseline (không làm lại) |
| 2.2 | `nwd` | Normalized Wasserstein Distance |
| 2.3 | `igwd` | Improved GWD (isotropic S, β=8) |
| 2.4 | `igwd_log_shape` | **NEW**: IGWD position (isotropic) + ALW log-ratio shape |
| 2.5 | `igwd_anisotropic_s` | **NEW**: ALW anisotropic position + IGWD Euclidean shape |
| 2.6 | `alw_full` | ALW hoàn chỉnh (anisotropic + reliability gate + Charbonnier) |
| 2.7 | `sa_alw_beta_only` | ALW + Scale-Adaptive β |
| 2.8 | `sa_alw_full` | ALW + SA-β + SA-pos-weight |

**Strategy**: 1 seed (42) trước để screening. Nếu metric tiềm năng, chạy thêm 2 seeds.

## Bug Fix: score_thresh khi eval

### Vấn đề

`collect_predictions()` trong `eval_utils.py` giữ `roi_heads.score_thresh = SCORE_THRESH_TRAIN = 0.05`. Tiny objects thường có score 0.01-0.04 → bị filter trước khi tính mAP, khiến kết quả bị đè ~5-10%.

### Fix

Hạ tạm `model.roi_heads.score_thresh = 0.001` khi eval, khôi phục sau khi xong. Apply trong `collect_predictions()` (line 291-295, 303).

### RFLA Pass 2 verification

Bug.txt đề xuất sửa `wn * beta` → `wn / beta` vì comment "mở rộng vùng". Tuy nhiên, paper RFLA Section 3.3 viết rõ: **"slight decay the effective radius erₙ by multiplying a stage factor β"** (β=0.9, Table 3). Tức là **thu nhỏ** 10% effective radius để tăng strictness ở Pass 2.

→ Code gốc `wn * beta` là **đúng paper**. Bug.txt **sai** ở điểm này. Không sửa.

## Code Changes

### metric/igwd.py — 2 metrics mới

**`compute_log_shape` (Phase 2.4)**: IGWD position term (isotropic `S = wp·hp + wt·ht`) + ALW log-ratio shape `[ln(wp/wt)]² + [ln(hp/ht)]²`. Tests whether log-ratio shape (scale-invariant) is better than linear `(dw²+dh²)/4`.

**`compute_anisotropic_s` (Phase 2.5)**: ALW anisotropic position `(Δx)²/Sx + (Δy)²/Sy` + IGWD Euclidean shape `(dw²+dh²)/4`. Tests whether anisotropic position normalization improves IGWD.

### metric/__init__.py

Register `igwd_log_shape` and `igwd_anisotropic_s` với display names.

### eval_utils.py

- Hạ `score_thresh` xuống 0.001 khi eval
- Fix `compute_precision_recall`: skip tile không có GT box (tránh `max(dim=0)` trên tensor rỗng)
- Fix double-count `total_fp`

### model.py

- Làm rõ comment `la_loss`: metric dùng trong RPN, box regression giữ Smooth-L1 vì `fastrcnn_loss` hoạt động trên deltas (không decode được thành box để tính metric)
- `la` = `la_loss` trong hiện tại — khác biệt duy nhất là metric trong RPN assigner

### scripts/train_frcnn_metric.py (NEW)

Training script cho Phase 2: nhận `--metric`, `--placement`, `--seed`. Tự động tính `reliability_thr` từ dataset nếu metric cần.

### scripts/run_phase2.py (NEW)

Run tất cả 7 metrics × 1 seed tuần tự. Có `--quick` (1 seed) và `--metric <name>` (single metric).

## Environment

- **Python**: 3.13.9 (system-wide, `C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\`)
- **PyTorch**: 2.11.0+cu128, CUDA 12.8
- **GPU**: RTX 5070 Ti (16GB)
- **torchvision**: 0.26.0+cu128
- **.venv**: Bị hỏng (link đến user cũ), không dùng. Global OK.

## Results (pending)

| Metric | seed42 mAP@50 |
|--------|:---:|
| nwd | |
| igwd | |
| igwd_log_shape | |
| igwd_anisotropic_s | |
| alw_full | |
| sa_alw_beta_only | |
| sa_alw_full | |

## Related Pages

- [[Phase 1 Baseline Setup - 2026-07-01]]
- [[Cascaded Routing Implementation Plan - 2026-07-01]]
- [[Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)]]
- [[Cascaded Uncertainty Routing]]
