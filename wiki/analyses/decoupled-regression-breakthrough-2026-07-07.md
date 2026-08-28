---
title: Decoupled Regression Breakthrough Plan - 2026-07-07
type: analysis
created: 2026-07-07
updated: 2026-07-07
sources: [wiki/research/deep-research-tiny-od-breakthroughs-2026-07-06.md, common/model.py, common/config.py, scripts/train_frcnn_metric.py]
tags: [breakthrough, regression-loss, ap75, decoupled, sa-alw, experiment-plan]
---

# Decoupled Regression Breakthrough Plan — 2026-07-07

## Chẩn đoán (đã xác nhận từ deep research)

Vanilla RFLA (assigner của chúng ta) đạt **AP75=18.8** trên AI-TOD-v2 với **Smooth-L1
regression chuẩn**. Chúng ta đang dùng Gaussian similarity (`1-exp(-β·D)`) cho CẢ
assignment VÀ regression loss (`common/model.py:242`). Hậu quả: AP75=0.03 (thấp hơn
4-8× so với vanilla RFLA). Nguyên nhân: Gaussian similarity bão hòa (gradient → 0)
trước khi box đạt tight localization — loss không còn tín hiệu để kéo box về đúng vị trí.

## Luận điểm (thesis)

> Gaussian/Wasserstein metrics tối ưu cho **assignment** (khớp anchor-GT với tiny objects
> có overlap thấp) nhưng **không phù hợp làm regression loss** (bão hòa, IoU-insensitive).
> Decouple hai vai trò: SA-ALW giữ assignment, regression dùng loss có gradient mạnh ở
> vùng overlap cao (Smooth-L1 / CIoU / DIoU).

## Hướng đột phá

### Experiment A: SA-ALW assignment + Smooth-L1 regression (mirror RFLA success)
- **Lý do:** Đây chính xác là pattern RFLA dùng để đạt AP75=18.8. SA-ALW assigner đã
  tốt hơn RFLA gốc (có anisotropy + log-shape + scale-adaptive). Kết hợp với Smooth-L1
  regression sẽ cho localization tốt hơn hẳn.
- **Thay đổi code:** `_metric_box_loss`: giữ nguyên decode pred_boxes/gt_boxes, nhưng
  thay `box_loss = (1-sim).mean()` bằng `smooth_l1_loss(box_regression_pos, targets_deltas)`.
- **Rủi ro:** Smooth-L1 có thể quá aggressive với micro objects (2-3px) → cần theo dõi
  AP_micro.
- **Success gate:** test AP@75 > 0.08, AP_micro không giảm quá 0.03.

### Experiment B: SA-ALW assignment + CIoU regression
- **Lý do:** CIoU có gradient mạnh ở vùng overlap cao (khác Gaussian), xét cả
  overlap + center distance + aspect ratio. Tốt cho AP@75.
- **Rủi ro:** Aspect ratio term có thể gây nhiễu cho micro objects (2-3px không có
  aspect ratio rõ ràng).
- **Success gate:** test AP@75 > 0.08, AP_micro không giảm quá 0.03.

### Experiment C: SA-ALW assignment + DIoU regression
- **Lý do:** DIoU giống CIoU nhưng bỏ aspect ratio term → ít nhiễu hơn cho tiny objects.
- **Success gate:** như trên.

### Experiment D (nếu A/B/C thành công): Multi-seed validation
- Chạy seed 42, 123, 2024 cho best variant → mean±std.
- So sánh với SA-ALW baseline multi-seed (0.5856±0.021).

## Implementation plan

### Code changes (common/model.py + common/config.py)

Thêm config flag `BOX_LOSS_TYPE` trong config.py:
- `"metric"` — Gaussian similarity (current, baseline)
- `"smooth_l1"` — standard Smooth-L1 on deltas
- `"ciou"` — CompleteIoU on decoded boxes
- `"diou"` — DistanceIoU on decoded boxes

Sửa `_metric_box_loss` để dispatch theo `BOX_LOSS_TYPE`.
Sửa `_wrap_roi_forward_for_metric_loss` và `build_model` để nhận tham số mới.
Sửa `train_frcnn_metric.py` để thêm flag `--box-loss`.

### Training config
- Metric: `sa_alw_full` (giữ nguyên)
- Placement: `la_loss` (metric trong RPN LA + RoI box loss — nhưng box loss sẽ thay đổi)
- Seed: 42
- Epochs: 20 (cosine schedule)
- Batch size: 4
- Full-image training (không patches)
- RPN proposals: 3000

### Output naming
- `sa_alw_full__smoothl1__seed42`
- `sa_alw_full__ciou__seed42`
- `sa_alw_full__diou__seed42`

## Execution order

1. Implement code changes (common/model.py, common/config.py)
2. Launch 3 experiments in parallel (A, B, C)
3. Monitor training (val mAP@50, val AP_micro per epoch)
4. Test-set evaluation cho best checkpoint mỗi experiment
5. So sánh với SA-ALW baseline (test mAP=0.6005, AP_micro=0.5091, AP@75≈0.03)
6. Nếu thành công → multi-seed + viết paper section

## Compute budget
- 3 experiments × 20 epochs × ~15 min/epoch = ~15h trên RTX 5070 Ti 16GB
- Có thể chạy tuần tự (VRAM đủ cho 1 experiment/lần)

## Success criteria
- **Primary:** test AP@75 > 0.08 (từ ~0.03 baseline) = breakthrough confirmed
- **Secondary:** AP_micro ≥ 0.47 (giữ được ≥90% baseline)
- **Stretch:** mAP(scale) > 0.60 (vượt baseline)
- **Paper-worthy:** Nếu bất kỳ variant nào đạt primary → có paper contribution rõ ràng

## Related Pages
- [[Deep Research: Tiny-OD Breakthroughs 2024–2026 & the AP@75 Diagnosis]]
- [[Decoupled DFL Regression Plan - 2026-07-06]]
- [[Phase 2-4 Results Summary]]
