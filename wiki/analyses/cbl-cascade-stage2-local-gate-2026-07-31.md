---
title: CBL Cascade Stage-2 Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [cvpr:2018-cascade-rcnn, common/model.py, runs/cbl_cascade_stage2_local_gate_best_ap75_valid_reload.json]
tags: [cbl, cascade, classification, localization, negative-result]
---

# CBL Cascade Stage-2 Local Gate - 2026-07-31

## Question

Can a higher-IoU second-stage classifier/regressor reduce the background
predictions introduced by shared-head trainable refinement?

## Method

1. Refine first-stage positive sampled RoIs with detached CBL boxes.
2. Keep sampled background RoIs at their original coordinates.
3. Re-match all stage-2 RoIs to GT at IoU `0.60`.
4. Subsample `256` RoIs per image with positive fraction up to `0.5`.
5. Train a cloned stage-2 MLP/predictor with softmax CE plus CBL loss.
6. At inference, preserve labels, refine boxes once, and average stage-1 and
   stage-2 class scores with weight `0.5`.

This is a bounded Cascade R-CNN adaptation: it adds re-matching and
classification but still operates only on final first-stage detections during
inference.

## Verification

- Synthetic matcher: exact-IoU proposal is foreground; IoU `0.49` is
  background at threshold `0.60`.
- Real batch-4 re-match: `140` foreground and `1,908` background before
  resampling; `140/884` after resampling, with both foreground classes.
- Stage-2 classifier and distribution predictor receive finite non-zero
  gradients.
- Shared-head and regression-only checkpoint behavior remains exact.
- Cascade inference/reload passes; batch-4 memory remains within the prior
  `3.716/3.811 GiB` allocated/reserved envelope.

## Two-Epoch Result

| Method | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Shared-head trainable refinement | **0.1269** | **0.3612** | **0.0572** | **0.2758** |
| Stage-2 cascade, score weight `0.50` | 0.1227 | 0.3537 | 0.0527 | 0.2684 |
| Stage-2 cascade, score weight `0.25` | 0.1236 | 0.3547 | 0.0546 | 0.2684 |
| Stage-2 cascade, score weight `0.00` | 0.1236 | 0.3529 | 0.0549 | 0.2682 |
| Stage-2 cascade, score weight `1.00` | 0.1181 | 0.3403 | 0.0525 | 0.2681 |

The second-stage score is harmful when trusted strongly. Preserving the
first-stage score is best for AP75, but the cascade still loses AP, AP50,
AP75, and AR to shared-head refinement.

## Decision

Negative promotion gate. Do not launch on Kaggle and do not sweep IoU or
classification-loss weights. Both regression-only specialization and the
bounded full cascade underperform, so the extra stage is not the next
high-value direction for this dataset.

The reloadable leader remains shared-head trainable iterative CBL EMA epoch
5 at AP/AP50/AP75/AR100=`0.1486/0.4030/0.0764/0.2949`.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__irh2c0.6cw1sw0.5__la_loss__seed42__cbl_cascade_stage2_local_gate/metrics.csv`
- `runs/cbl_cascade_stage2_local_gate_best_ap75_valid_reload.json`
- `runs/cbl_cascade_stage2_local_gate_best_ap75_scorew00_valid.json`
- `runs/cbl_cascade_stage2_local_gate_best_ap75_scorew025_valid.json`
- `runs/cbl_cascade_stage2_local_gate_best_ap75_scorew10_valid.json`
- branch `cbl-cascade-stage2-20260731`

## Related Pages

- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[Stage-Specific CBL Refinement Local Gate - 2026-07-31]]
- [[Iterative CBL Refinement Gate - 2026-07-31]]
- [[Wiki Log]]
