---
title: "Action Plan: Post Deep-Research Execution Roadmap"
type: analysis
created: 2026-06-05
updated: 2026-06-05
sources:
  - wiki/research/deep-research-architecture-and-training.md
tags: [action-plan, tiny-object-detection, roadmap]
---

# Action Plan: Post Deep-Research Execution Roadmap

> **Tình trạng hiện tại**: Best mAP(scale)=0.5770, AP_micro=0.2776, **AP@75=0.0428** (bottleneck chung).
> Root cause đã xác định: Gaussian similarity loss `1−exp(−β·D_H)` **không nhạy với IoU** — không phạt lỗi localization nhỏ.

---

## Phase 1 — Phá vỡ bottleneck AP@75 (Ưu tiên cao nhất)

### Experiment 15: DIoU Scheduled Regression
**File**: `working/code/15_p2_diou_scheduled.ipynb`

| Thay đổi | Chi tiết |
|---|---|
| **Loss regression mới** | `L_reg = (1−S_H) + γ_eff · L_DIoU` (dual objective) |
| **Scheduled ramp-up** | epoch 1-3: warmup Smooth-L1 → epoch 3-8: ramp γ từ 0→0.3 → epoch 8+: γ=0.3 full |
| **Tại sao DIoU?** | Center penalty cho gradient ngay cả khi IoU=0 (rất phổ biến với object <8px) |
| **Baseline** | Clean P2F (như NB12, không RoIAlign 14, không conv head) |
| **Kỳ vọng** | AP@75: 0.04 → 0.06-0.08, mAP(scale) ≥ 0.57 |

**Tại sao γ=0.3 chứ không phải 1.0?** NB10 đã thất bại với GAMMA_FINE=1.0 từ epoch 1 — DIoU quá mạnh sớm gây gradient nhiễu khi proposals còn rất thô.

### Experiment 14: RoIAlign 14×14 + ConvHead (đã tạo)
**File**: `working/code/14_p2_roialign14_convhead.ipynb`

| Thay đổi | Chi tiết |
|---|---|
| RoIAlign output_size | 7→14, sampling_ratio 2→4 |
| Box head | ConvBoxHead: Conv3×3 → Conv3×3-stride2 → FC6 → FC7 (14→7 learned downsample) |
| **Kỳ vọng** | AP_micro +1-3%, AP@75 +1-2% (cần kết hợp với DIoU để maximize) |

### Experiment 16: Combined Winner (DIoU + RoIAlign 14×14)
**File**: `working/code/16_p2_roialign14_diou_combined.ipynb`

Kết hợp cả 2 thay đổi từ NB14 + NB15:
- RoIAlign 14×14 + ConvHead (cho spatial resolution)
- DIoU scheduled regression (cho precision loss signal)

**Kỳ vọng**: AP@75 > 0.06, mAP(scale) ≥ 0.58-0.60

---

## Phase 2 — Stack micro gains (Sau khi Phase 1 có kết quả)

### Experiment 17: Scale-Adaptive k Tuning
- Dùng winner Phase 1
- Tune k_micro=6 (giảm từ 9), k_tiny=5, k_other=3
- Kỳ vọng: AP_micro +1-2%

### Experiment 18: Multi-Resolution Training
- Random training resolution 640-1024 (thay vì fixed 512)
- Dễ implement, lợi ích nhỏ nhưng consistent
- Kỳ vọng: mAP +0.5-1%

---

## Phase 3 — Advanced (Chỉ nếu Phase 1-2 plateau)

| Experiment | Ý tưởng | Complexity | Kỳ vọng |
|---|---|---|---|
| 2-stage Cascade | IoU threshold [0.4, 0.55] với Gaussian distance matching | Cao | AP@75 +1-3% |
| Copy-Paste augmentation | Paste micro objects vào images khác | Trung bình | AP_micro +2-4% |
| CIoU thay DIoU | Nếu aspect ratio precision cần thiết | Thấp | Marginal |

---

## Nguyên tắc thực thi

1. **Một biến tại một thời điểm** — NB15 chỉ thay loss, NB16 kết hợp SAU KHI cả NB14 và NB15 có kết quả
2. **Fix loss trước architecture** — Không có kiến trúc nào fix được AP@75 nếu loss không reward precision
3. **RoIAlign + DIoU bổ sung nhau** — RoIAlign cho head thêm thông tin spatial, DIoU dạy head dùng thông tin đó cho precise boxes
4. **Label assignment đã gần tối ưu** — RFLA + Gaussian distance + scale-adaptive k đã tốt, bottleneck giờ ở head/loss

---

## Thứ tự chạy trên Kaggle

```
1. Chạy NB14 (RoIAlign 14×14 + ConvHead)     ← đã tạo, chạy ngay
2. Chạy NB15 (DIoU scheduled regression)       ← tạo tiếp theo
3. So sánh kết quả NB14 vs NB15
4. Chạy NB16 (combined)                        ← tạo sau khi có kết quả
5. Phase 2 experiments nếu Phase 1 thành công
```

---

*Document created: 2026-06-05. Based on deep research synthesis from `wiki/research/deep-research-architecture-and-training.md`.*