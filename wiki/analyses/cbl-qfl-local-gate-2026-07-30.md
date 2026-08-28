---
title: CBL Quality Focal Loss Local Gate - 2026-07-30
type: analysis
created: 2026-07-30
updated: 2026-07-30
sources: [neurips:2020-gfl, open-mmlab:mmdetection-gfl, common/model.py, runs/sa_alw_full__cbl__qflb2__la_loss__seed42__cbl_qfl_local_gate/metrics.csv]
tags: [cbl, qfl, gfl, classification, localization-quality, ap75, negative-result]
---

# CBL Quality Focal Loss Local Gate - 2026-07-30

## Question

Does replacing the RoI softmax classifier with a joint class-IoU score improve
the ranking of the stronger CBL boxes without repeating the failed standalone
quality-head design?

## Research Basis

Generalized Focal Loss (NeurIPS 2020) identifies a training-inference mismatch
when classification and localization-quality scores are trained independently
but multiplied at inference. QFL instead uses a foreground class vector whose
target at the ground-truth class is the predicted-box IoU and whose other
targets, including all background proposals, are zero. Inference uses the
sigmoid joint score directly.

The local implementation follows the paper's QFL equation and the official
OpenMMLab loss:

- foreground classes use independent sigmoid logits;
- positive target = detached paired IoU of the decoded CBL box and GT;
- negatives and non-target classes use target zero;
- modulating factor = `abs(target - sigmoid(logit)) ** beta`;
- `beta=2`, the paper default;
- no auxiliary quality head and no score multiplication.

This is a bounded two-stage adaptation. GFL was designed and validated as a
dense one-stage detector, so this experiment is not a paper-faithful detector
reproduction.

## Technical Verification

- formula matched an independent tensor implementation;
- background logit received zero gradient because it is excluded from the
  foreground sigmoid vector;
- CUDA forward/backward and inference passed;
- checkpoint save/reload reproduced inference scores;
- standard CBL smoke and checkpoint-source tests still passed.

## Validation Gate

Both runs used seed 42, CBL, SA-ALW label assignment, copy-paste, tiny-tile
oversampling, no EMA, and zero workers.

| Model | Epoch | COCO AP | AP50 | AP75 | AR100 | weighted class-aware AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| CBL standard | 1 | **0.1145** | **0.3334** | **0.0454** | **0.2692** | - | - |
| CBL + QFL | 1 | 0.0853 | 0.2382 | 0.0366 | 0.2580 | 0.3451 | 0.1963 |
| CBL standard | 2 | **0.1200** | **0.3523** | **0.0471** | **0.2759** | **0.4938** | **0.3515** |
| CBL + QFL | 2 | 0.0965 | 0.2736 | 0.0418 | 0.2561 | 0.3914 | 0.2317 |

Independent reload of the epoch-2 best checkpoint reproduced
AP/AP50/AP75/AR100=`0.0965/0.2735/0.0419/0.2562`.

The first process slowed after epoch-1 validation and was terminated by the
one-hour command timeout. Resuming epoch 2 from `last.pt` in a fresh process
completed normally in 764 seconds. This lifecycle issue does not change the
negative metric comparison.

## Decision

Negative local performance gate. Do not launch CBL+QFL on Kaggle and do not
evaluate it on the locked test set. The exact joint-score idea is better founded
than the old separate quality head, but QFL's dense one-stage sigmoid objective
does not transfer cleanly to the balanced, sampled RoI classifier in this
project. It reduces total AP, AP75, recall, and especially micro class-aware AP.

Do not tune `beta` blindly. Revisit score-localization coupling only with a
two-stage-specific ranking/calibration objective or a distribution-guided
predictor that is validated without direct score multiplication.

## Artifacts

- `runs/sa_alw_full__cbl__qflb2__la_loss__seed42__cbl_qfl_local_gate/metrics.csv`
- `runs/cbl_qfl_local_gate_best_ap75_valid_reload.json`
- `scripts/test_quality_focal_cbl.py`

## Related Pages

- [[Confidence-Driven Localization Local Gate - 2026-07-30]]
- [[Decoupled DFL Regression Plan - 2026-07-06]]
- [[Deep Research: Tiny-OD Breakthroughs 2024–2026 & the AP@75 Diagnosis]]
