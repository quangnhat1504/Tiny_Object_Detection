---
title: Iterative CBL Refinement Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [cvpr:2018-cascade-rcnn, common/model.py, runs/kaggle_cbl_ema8_best_ap75_valid_reload.json, runs/cbl_ema8_ep5_iterative_refine1_score030_valid.json]
tags: [cbl, iterative-refinement, inference, localization, ap75, positive-result]
---

# Iterative CBL Refinement Gate - 2026-07-31

## Question

Does reapplying the trained CBL regressor to its final detections improve
strict localization enough to justify a trainable cascade stage?

## Method

Cascade R-CNN motivates sequential box refinement, but a full trainable
cascade introduces stage heads, matching thresholds, and a new training
schedule. This gate first tests the core mechanism without retraining:

1. Run normal CBL inference, including thresholding and class-wise NMS.
2. RoIAlign the resulting maximum 100 detections.
3. Reapply the same trained CBL head.
4. Select the class-specific delta for each existing label.
5. Update only boxes whose original class score reaches a configured
   threshold; preserve every label and score.
6. Clip boxes and rerun class-wise NMS.

The feature extractor, weights, labels, scores, candidate budget, and CBL
decoder do not change. `cbl_refine_steps=0` remains the default, so existing
checkpoint behavior is unchanged.

## Verification

- CUDA smoke covers zero/two-step inference with finite boxes and at most 100
  detections per image.
- Training, standard inference, and checkpoint reload still pass.
- A step blend sweep `0.5/0.75/1.0/1.25` found the full update best for AP75.
- A score gate sweep `0/0.02/0.05/0.10/0.20/0.30/0.50` identified `0.20` as
  AP75-first and `0.30` as the better AP/AR balance.
- Dummy `640x800`, batch-1 speed was `49.1 FPS` at zero steps, `43.4 FPS` at
  one step, and `39.8 FPS` at two steps.

## Leader Validation

All rows use the reloadable CBL-EMA epoch-5 checkpoint on the full validation
set.

| Refinement | AP | AP50 | AP75 | AR100 | weighted class-aware AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| 0 steps | 0.1409 | 0.3891 | 0.0665 | **0.2947** | 0.5270 | 0.3697 |
| 1 step, all scores | 0.1479 | 0.4032 | 0.0745 | 0.2884 | 0.5456 | 0.3706 |
| 1 step, score >= 0.20 | 0.1479 | 0.4036 | **0.0747** | 0.2895 | **0.5456** | 0.3718 |
| 1 step, score >= 0.30 | **0.1481** | 0.4038 | 0.0746 | 0.2920 | 0.5455 | **0.3792** |
| 2 steps, all scores | **0.1481** | **0.4054** | 0.0731 | 0.2849 | 0.5511 | 0.3661 |

The balanced one-step `score>=0.30` profile improves AP by `0.0072`, AP50 by
`0.0147`, and AP75 by `0.0081` while recovering most of the AR lost by
refining every detection. The one-step `score>=0.20` profile is the AP75-first
alternative.

## Cross-Checkpoint Check

The same one-step mechanism improved AP and AP75 on weaker raw checkpoints:

| Checkpoint | Setting | AP | AP50 | AP75 | AR100 |
|---|---|---:|---:|---:|---:|
| CBL raw epoch 5 | base | 0.1277 | 0.3659 | 0.0554 | 0.2768 |
| CBL raw epoch 5 | refine, score >= 0.30 | **0.1308** | **0.3755** | **0.0572** | 0.2707 |
| CBL raw local epoch 2 | base | 0.1199 | 0.3523 | 0.0467 | 0.2759 |
| CBL raw local epoch 2 | refine all | **0.1206** | **0.3595** | **0.0471** | 0.2643 |

The gain is largest on the better EMA checkpoint, but its direction is not
checkpoint-specific. AR consistently declines, so selective refinement is
required.

## Error Audit

At score threshold `0.05` and top 100 detections:

| Metric | EMA base | Refine score >= 0.30 | Change |
|---|---:|---:|---:|
| TP@75 | 1,627 | 1,661 | +34 |
| recall75 | 0.1966 | 0.2007 | +0.0041 |
| precision75 greedy | 0.0188 | 0.0194 | +0.0006 |
| localization FP, IoU 0.50-0.75 | 6,981 | 5,901 | -1,080 (-15.5%) |
| localization FP, IoU 0.25-0.50 | 16,221 | 14,072 | -2,149 (-13.2%) |

This confirms that the AP75 gain reflects tighter boxes rather than score
recalibration.

## Decision

Positive full-validation gate. Use one step with score threshold `0.30` as the
balanced validation leader and threshold `0.20` only for an explicitly
AP75-first profile.

No Kaggle training is needed because the method reuses an existing
checkpoint. Do not reopen the locked test set for this CBL-family iteration:
the family already consumed its frozen test gate. Require an external dataset,
new blind holdout, or separately budgeted final evaluation before making a
test-generalization claim.

The strong localization signal justifies the next research step: a lightweight
trainable refinement stage. It must preserve the current score/label path,
target already-refined proposals, and include an AR-preservation gate.

## Artifacts

- `runs/cbl_ema8_ep5_iterative_refine1_score020_valid.json`
- `runs/cbl_ema8_ep5_iterative_refine1_score030_valid.json`
- `runs/cbl_ema8_ep5_iterative_refine2_valid.json`
- `runs/cbl_full_raw_ep5_iterative_refine1_score030_valid.json`
- `runs/ap75_analysis_cbl_ema8_ep5_base_valid/summary.json`
- `runs/ap75_analysis_cbl_ema8_ep5_refine1_score030_valid/summary.json`
- branch `cbl-iterative-refine-20260731`, commit `63ecc83`

## Related Pages

- [[CBL EMA Recovery Audit - 2026-07-30]]
- [[Double-Head CBL Local Gate - 2026-07-31]]
- [[Confidence-Driven Localization Local Gate - 2026-07-30]]
- [[Wiki Log]]
