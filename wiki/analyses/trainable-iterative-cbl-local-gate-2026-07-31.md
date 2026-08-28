---
title: Trainable Iterative CBL Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [cvpr:2018-cascade-rcnn, common/model.py, runs/cbl_iterative_train_local_gate_best_ap75_valid_reload.json, runs/kaggle_cbl_iterative_train_ema8_best_ap75_valid_reload.json]
tags: [cbl, iterative-refinement, training, localization, ap75, positive-result, kaggle]
---

# Trainable Iterative CBL Local Gate - 2026-07-31

## Question

Can the existing CBL RoI head learn from its own once-refined proposals,
without adding a second parameter-heavy cascade head?

## Method

For each sampled positive RoI during training:

1. Decode the first-pass class-specific CBL distribution.
2. Detach the resulting proposal coordinates.
3. RoIAlign the refined proposal and run the shared box head/predictor again.
4. Re-encode the same GT box relative to that refined proposal.
5. Apply the existing CBL localization loss with weight `0.5`.

The detach boundary avoids an unstable end-to-end coordinate path. The
classification loss remains first-pass only. At inference, one shared-head
refinement step is applied only to detections with score at least `0.30`.

## Technical Gate

- CUDA forward/backward produced finite `loss_box_refine=1.5592`.
- Gradients reached both `box_head.fc7` and the CBL distribution predictor.
- Batch size 4 peaked at `3.664 GiB` allocated and `3.756 GiB` reserved.
- Refined targets remained inside the configured `[-5, 5]` CBL grid.
- Checkpoint reconstruction preserved inference output exactly.

## Two-Epoch Validation

| Setting | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Standard CBL, no refinement | 0.1199 | 0.3523 | 0.0467 | 0.2759 |
| Standard CBL + inference refinement | 0.1206 | 0.3595 | 0.0471 | 0.2643 |
| Trainable refinement, inference disabled | 0.1195 | 0.3438 | 0.0503 | **0.2798** |
| Trainable + inference refinement | **0.1269** | **0.3612** | **0.0572** | 0.2758 |

The auxiliary training target alone improves AP75 and AR, while the full
train/inference path improves AP and AP75 without the AR collapse observed
when applying inference refinement to the standard local checkpoint.
Independent reload reproduced AP/AP50/AP75/AR100 =
`0.1269/0.3612/0.0572/0.2758`.

## Kaggle EMA8 Audit

The private run completed all eight epochs and downloaded `metrics.csv`,
`best.pt`, `best_ap75.pt`, `best_coco_ap.pt`, and `last.pt`. All three best
checkpoints contain the evaluated EMA epoch-5 model and are reloadable.

| Checkpoint/profile | AP | AP50 | AP75 | AR100 | weighted class-aware AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| CBL EMA epoch 5, no refinement | 0.1409 | 0.3891 | 0.0665 | 0.2947 | 0.5270 | 0.3697 |
| CBL EMA + inference-only refinement | 0.1481 | **0.4038** | 0.0746 | 0.2920 | 0.5455 | 0.3792 |
| Trainable iterative EMA, inference disabled | 0.1409 | 0.3836 | 0.0678 | **0.2957** | 0.5166 | 0.3707 |
| Trainable iterative EMA + score `>=0.30` | **0.1486** | 0.4030 | **0.0764** | 0.2949 | **0.5395** | **0.3804** |

Independent reload reproduced AP/AP50/AP75/AR100 =
`0.1486/0.4030/0.0764/0.2949`. Score threshold `0.20` tied AP75 but reduced
AP and AR to `0.1484/0.2921`, so `0.30` remains the balanced profile.

The peak is epoch 5. Epoch 8 declines to AP/AP75/AR100 =
`0.1341/0.0620/0.2777`; use the audited best checkpoint, not `last.pt`.

Against the inference-only leader at score threshold `0.05` and top 100, the
new model raises TP@75 from `1,661` to `1,702` and recall75 from `0.2007` to
`0.2057`. Micro/tiny TP@75 improve by `+11/+41`, while localization FP in the
IoU `0.50-0.75` band is essentially flat (`5,901 -> 5,893`). Background
predictions increase, so the next experiment must preserve the new tiny-object
recall without adding indiscriminate confidence.

## Decision

Positive full-budget gate and the current reloadable validation leader. The
method improves AP and AP75 over inference-only refinement while restoring
AR. It does not justify another locked-test look because the CBL family
already consumed its test budget.

Kernel and pinned source:

- kernel `quangnhtng/tod-cbl-itrain-ema8-20260731`
- branch `cbl-iterative-refine-20260731`
- commit `21de4e498faff4859a0ab2055e2a126fd4cf402d`
- run `cbl_iterative_train_ema8`

Subsequent stage-specific and classified-cascade experiments both failed.
Paired refinement analysis instead found that this trainable shared head can
benefit from additional inference depth: three ungated passes produce the new
strict-validation leader at AP/AP50/AP75/AR100 =
`0.1501/0.4074/0.0774/0.2934`. Gating the extra passes at a normalized
predicted size equivalent to 12 px produces the overall-AP leader
`0.1504/0.4081/0.0772/0.2946`. See
[[CBL Refinement Consistency and Depth Gate - 2026-07-31]].

## Refinement-Loss Weight Ablation

A lower auxiliary-loss weight (`0.25` instead of `0.50`) was tested with the
same seed, raw-weight training, augmentation, inference refinement, and
two-epoch local budget.

| Weight / epoch | AP | AP50 | AP75 | AR100 | weighted class-aware AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| 0.25 / 1 | 0.1210 | 0.3415 | **0.0556** | 0.2661 | 0.4892 | 0.3285 |
| 0.25 / 2 | 0.1214 | 0.3469 | 0.0507 | 0.2645 | 0.4884 | 0.2972 |
| 0.50 / 1 | 0.1161 | 0.3328 | 0.0498 | 0.2613 | - | - |
| 0.50 / 2 | **0.1269** | **0.3612** | **0.0572** | **0.2758** | **0.5084** | **0.3578** |

Weight `0.25` accelerates the first epoch but its strict-localization result
declines in epoch 2. Independent reload of its epoch-1 `best_ap75.pt`
reproduced AP/AP50/AP75/AR100=`0.1209/0.3414/0.0556/0.2661`. It remains below
weight `0.50` after the fair two-epoch budget on every COCO metric and both
class-aware summaries. This is a negative ablation: retain `0.50` and do not
launch the lower weight on Kaggle.

## Artifacts

- `runs/cbl_iterative_train_local_gate_best_ap75_valid_reload.json`
- `runs/cbl_iterative_train_local_gate_best_ap75_no_inference_refine_valid.json`
- `runs/ap75_analysis_cbl_iterative_train_local_gate_valid/summary.json`
- `runs/kaggle_cbl_iterative_train_ema8_best_ap75_valid_reload.json`
- `runs/kaggle_cbl_iterative_train_ema8_best_ap75_no_inference_refine_valid.json`
- `runs/kaggle_cbl_iterative_train_ema8_best_ap75_refine_score020_valid.json`
- `runs/ap75_analysis_cbl_iterative_train_ema8_ep5_valid/summary.json`
- `runs/cbl_iterative_train_w025_local_gate_best_ap75_valid_reload.json`
- `.runtime/kaggle/cbl_iterative_train_ema8/audit.json`
- `.runtime/kaggle/cbl_iterative_train_ema8/state.json`

## Related Pages

- [[Iterative CBL Refinement Gate - 2026-07-31]]
- [[CBL EMA Recovery Audit - 2026-07-30]]
- [[Confidence-Driven Localization Local Gate - 2026-07-30]]
- [[Wiki Log]]
