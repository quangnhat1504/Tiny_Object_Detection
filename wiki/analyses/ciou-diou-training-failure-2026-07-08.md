---
title: CIoU/DIoU Decoupled Regression Training Failure — 2026-07-08
type: analysis
created: 2026-07-08
updated: 2026-07-08
sources: [common/model.py, common/train_utils.py, common/eval_utils.py, scripts/train_frcnn_metric.py]
tags: [ciou, diou, crash, bug, cuda, NaN, regression-loss, failed-experiment, ap75]
---

# CIoU/DIoU Decoupled Regression Training Failure — 2026-07-08

## Summary

CIoU/DIoU box regression (SA-ALW assignment + CIoU/DIoU loss) is **not trainable** on this dataset with the current 16GB VRAM + autocast setup. Four consecutive attempts all failed with CUDA-level crashes or silent NaN poisoning. The experiment is **blocked** until either (a) gradient clipping for IoU losses is added, (b) full float32 training is used (needs >24GB VRAM), or (c) the approach is redesigned.

## Timeline of Failures

### Attempt 1 — CIoU seed 42 (fresh, original code)
- Epoch 1-6: Normal training, loss ~1.0-1.4
- Epoch 7: Best val mAP@50=0.3933, coco_AP75=0.0600 (better than SA-ALW baseline 0.0515!)
- Epoch 8: Loss spikes 1.01→2.91, val mAP collapses 0.3933→0.3172
- Epoch 9: Loss explodes 2.91→17.76→68.36, crash on `torch.isfinite(int)` at step 975

**Root cause:** CIoU aspect-ratio term produces NaN on boxes < 4px in float16 (width→0 → atan division by zero). `sum()` over all-NaN tensors returns Python `int(0)`, then `torch.isfinite(0)` crashes (int is not Tensor).

### Attempt 2 — CIoU seed 42 (resume with fix #1)
- Fix: Guard `isinstance(loss, int)` in train_utils.py line 114
- Epoch 9 resumed: loss=5.1 (3x normal), eval produces all zeros — optimizer state already poisoned from Attempt 1
- Crash on `scaler.scale(loss).backward()` — CUDA-level RuntimeError: "variable modified by inplace operation"

**Root cause:** `pred_boxes[:, 2] = pred_boxes[:, 2].clamp(...)` — inplace assignment on tensor that requires grad, breaks autograd graph.

### Attempt 3 — CIoU seed 42 (resume with fixes #1+#2)
- Fix: Replace inplace clamp with `torch.stack` (non-inplace)
- Epoch 9: Training continues but val_loss=44, mAP=0.0000 — model already dead from Attempt 1's poisoned optimizer state
- Epoch 10: val_loss=251 → confirmed model irrecoverable

**Root cause:** SGD momentum buffers accumulate NaN gradients from Attempt 1-2. Resume is impossible — need clean restart.

### Attempt 4 — CIoU seed 42 (clean restart with all fixes)
- Deleted old checkpoints, started fresh
- Epoch 1-7: Normal training
- Epoch 8 at step 2128/2214 (96%): **CUDA C++ crash** — `c10::cuda::c10_cuda_check_implementation` → `CUDAPluggableAllocator::enable` → `StorageImpl::~StorageImpl`

**Crash stack trace (extracted):**
```
Unhandled exception caught in c10/util/AbortHandler.h
  torch_python.dll!THPCppFunction_requires_grad
  c10_cuda.dll!c10_cuda_check_implementation
  torch_cuda.dll!CUDAPluggableAllocator::enable
  c10.dll!StorageImpl::~StorageImpl
  c10.dll!TensorImpl::~TensorImpl
```

**Root cause:** CUDA runtime error during tensor deallocation. Likely OOM + memory fragmentation at epoch 8 after ~7.5h training, or float16 underflow producing an illegal CUDA memory access.

## All Bugs Found and Fixed

| # | File | Line | Bug | Severity | Status |
|---|------|------|-----|----------|--------|
| 1 | `train_utils.py` | 114 | `sum([])` → `int(0)` → `torch.isfinite(int)` TypeError | Crash | Fixed |
| 2 | `train_utils.py` | 131 | `torch.isfinite(loss_dict[k])` on non-Tensor value → TypeError | Crash | Fixed |
| 3 | `eval_utils.py` | 284 | `sum([])` → 0 silently (wrong val_loss when all losses NaN) | Silent corruption | Fixed |
| 4 | `model.py` | 289 | Inplace clamp on autograd tensor → RuntimeError | Crash | Fixed |
| 5 | `model.py` | 287 | CIoU NaN on zero-area boxes in float16 (atan term divide-by-zero) | NaN poisoning | Fixed* |
| 6 | `test_eval.py` | 54-62 | `parts[2]` parsing fails on compound names like `sa_alw_full__ciou__la_loss__seed42` | ValueError | Not fixed (replaced by test_coco_eval_single.py) |

*Fix #5: clamp min width/height to 2px, clamp CIoU loss max to 5.0, NaN guard → 0.0. But CUDA hard crash still occurs — NaN propagates through autograd faster than guard can intercept.

## Why CIoU/DIoU Cannot Train on This Dataset

**Float16 precision + tiny boxes (2-3px) + CIoU aspect-ratio term = unbounded overflow.**

CIoU formula: `L_CIoU = 1 - IoU + ρ²/c² + α·v` where `v = 4/π²·(atan(w_gt/h_gt) - atan(w_pred/h_pred))²`

For a 2×3px box: `atan(2/3) ≈ 0.588`, prediction near 0×0px: `atan(0) = 0`, so `v ≈ 4/π²·0.346 ≈ 0.140`, `α = v/(1-IoU+v)` with IoU≈0 → α≈0.123. Seems safe. **BUT** in float16: width underflow to 0 → `w_pred=0` → `atan(0/0)` = undefined → NaN.

The 2px clamp should prevent this, but the NaN may originate **inside** `complete_box_iou_loss`'s internal computation before our clamp takes effect — specifically in the decode path where `pred_boxes` originates from `box_coder.decode(box_regression_pos, proposals_pos)`. If any proposal has zero width/height, the decoded box can have `x2 < x1`.

**Three blocking issues:**
1. NaN propagation in float16 is faster than any Python guard can intercept → CUDA crash
2. Full float32 training needs ~24GB+ VRAM (current RTX 5070 Ti = 16GB)
3. SGD + autocast is unstable for IoU-based losses on tiny boxes — even if NaN is caught, the exploding gradient destabilizes training

## Recommendation

**Do not pursue CIoU/DIoU decoupled regression on this hardware.** Three alternatives:

1. **Smooth-L1 regression (Experiment A)** — No aspect-ratio term, no IoU computation, numerically stable in float16. The SA-ALW + smooth_l1 run reached val mAP@50=0.311 at epoch 4 (vs SA-ALW baseline 0.24 at same epoch). Already has code support. **This is the most promising path.**

2. **DFL regression head** — Distribution-based, no explicit IoU loss, no aspect-ratio. Needs new head architecture (~2-3 days implementation). Higher engineering cost but theoretically sound.

3. **Keep SA-ALW as-is, focus on WBF/ensemble** — Accept the AP75 ceiling from metric regression loss, improve via post-processing (SmartWBF already at mAP=0.6070).

## Related Pages
- [[Decoupled Regression Breakthrough Plan - 2026-07-07]]
- [[Decoupled DFL Regression Plan - 2026-07-06]]
- [[Phase 2-4 Results Summary]]
