---
title: Double-Head CBL Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [cvpr:2020-double-head-rcnn, mmdetection:v2.28.2-double-head, common/model.py, runs/sa_alw_full__cbl__dh4s1.3__la_loss__seed42__cbl_double_head_local_gate/metrics.csv]
tags: [cbl, double-head, localization, roi-head, ap75, negative-result]
---

# Double-Head CBL Local Gate - 2026-07-31

## Question

Can a dedicated convolutional RoI regression branch improve strict
localization after CBL-EMA made classification ranking reasonably healthy?

## Research Basis

Double-Head R-CNN separates the representations used for classification and
box regression. Its official Faster R-CNN implementation uses a two-FC
classification branch and a convolutional regression branch operating on
RoIs enlarged by a factor of `1.3`.

The bounded project adaptation preserves the pretrained torchvision
`TwoMLPHead` for classification and changes only CBL regression:

- normal RoIs feed the existing two-FC classification features;
- center-preserving, image-clipped `1.3x` RoIs feed regression;
- the regression branch uses a residual `256 -> 1024` projection, four
  ResNet bottlenecks, global average pooling, and the existing class-specific
  CBL distribution output;
- CBL targets/loss/decoding, HLA, RPN, data, augmentation, optimizer, seed,
  sampling, and inference thresholds remain unchanged.

The model grows from `41.373M` to `46.961M` parameters. This is a scoped
Double-Head/CBL hybrid rather than a full MMDetection reproduction.

## Technical Verification

- RoI enlargement preserves centers and clips to image bounds.
- CUDA AMP forward/backward gives finite non-zero gradients through the
  classifier, residual projection, final bottleneck, and distribution layer.
- Inference and checkpoint reload reproduce identical boxes and scores in the
  focused smoke test.
- Standard CBL forward/backward/inference/reload still passes.
- Batch size 4 fits, but Windows CUDA cache fragmentation eventually caused
  WDDM paging. An opt-in `TOD_EMPTY_CACHE_EVERY` control was added with default
  zero. Setting it to `1` for this local gate kept epoch-2 throughput near
  `2.8-3.0` batches/s without changing model computation or optimizer state.

## Validation Gate

Both candidates use seed 42, raw weights, no workers, two epochs, CBL,
copy-paste, and tiny-tile oversampling.

| Model | Epoch | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|---:|
| CBL standard | 1 | **0.1145** | **0.3334** | **0.0454** | **0.2692** |
| CBL + Double-Head | 1 | 0.0896 | 0.2796 | 0.0296 | 0.2418 |
| CBL standard | 2 | **0.1200** | **0.3523** | **0.0471** | **0.2759** |
| CBL + Double-Head | 2 | 0.1050 | 0.3179 | 0.0354 | 0.2519 |

Independent reload of Double-Head epoch 2 produced
AP/AP50/AP75/AR100=`0.1047/0.3176/0.0365/0.2513`. Its weighted and micro
class-aware scale AP were `0.4666/0.2788`, below standard CBL
`0.4938/0.3515`.

The AP75 error audit measured:

- recall75 `0.1524`;
- `7,611` same-class predictions localized between IoU 0.50 and 0.75;
- `16,312` predictions between IoU 0.25 and 0.50;
- class-agnostic AP75 `0.0435`, still too low to explain the deficit mainly
  by classification.

Adding a much larger regression representation therefore did not repair the
strict-localization bottleneck.

## Decision

Negative local gate. Do not launch Double-Head CBL on Kaggle and do not open
the locked test set. Do not sweep RoI scale or bottleneck count: the
paper-default architecture already loses AP, AP75, AR100, and micro
class-aware AP while adding parameters and latency.

The next experiment should change the refinement mechanism, not merely add
regression capacity. Prefer a lightweight iterative box-refinement stage with
explicitly controlled proposal matching before considering a full Cascade
R-CNN.

## Artifacts

- `runs/sa_alw_full__cbl__dh4s1.3__la_loss__seed42__cbl_double_head_local_gate/metrics.csv`
- `runs/cbl_double_head_local_gate_best_ap75_valid_reload.json`
- `runs/ap75_analysis_cbl_double_head_local_gate_valid/summary.json`
- `scripts/test_double_head_cbl.py`
- branch `cbl-double-head-20260730`, commit `272b4f0`

## Related Pages

- [[CBL EMA Recovery Audit - 2026-07-30]]
- [[CBL Rank and Sort Local Gate - 2026-07-30]]
- [[Confidence-Driven Localization Local Gate - 2026-07-30]]
- [[Wiki Log]]
