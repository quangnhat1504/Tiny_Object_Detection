---
title: CBL RPN IoU Quality EMA8 Audit - 2026-08-01
type: analysis
created: 2026-08-01
updated: 2026-08-01
sources: [eccv:2020-paa, common/model.py, runs/cbl_iterative_rpn_iou_detached_ema8_best_ap75_valid_reload.json, runs/rpn_proposal_recall_cbl_iterative_rpn_iou_detached_ema8_ep4_valid.json, runs/rpn_proposal_recall_cbl_iterative_train_ema8_ep5_valid.json]
tags: [cbl, rpn, iou-prediction, paa, ema, proposal-recall, negative-result]
---

# CBL RPN IoU Quality EMA8 Audit - 2026-08-01

## Question

Can separate foreground-presence and localization-quality signals improve RPN
proposal ranking and trainable iterative-CBL AP75 without losing micro/small
proposal coverage?

## Method

The first implementation follows the bounded parts of
[PAA](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5041_ECCV_2020_paper.php):

1. add a `3x3` RPN IoU predictor;
2. train it on positive anchors with decoded proposal-to-GT IoU targets;
3. detach predicted boxes when forming IoU targets;
4. weight the positive-only BCE by `0.5`;
5. rank proposals with `sqrt(presence * predicted_iou)`.

This is not full PAA. The project retains SA-ALW assignment, its nine-anchor
RPN contract, and Faster R-CNN RoI training. It does not use PAA's
probabilistic anchor assignment.

Focused CUDA tests cover exact fusion math, positive/empty-target loss,
finite gradients, no IoU-target gradient into bbox regression, proposal
reload, and legacy RPN paths.

## Shared-Tower Gate

The initial IoU predictor consumed the standard shared RPN tower feature. Its
two-epoch raw gate was unstable:

| Epoch | AP | AP50 | AP75 | AR100 |
|---:|---:|---:|---:|---:|
| 1 | 0.1187 | 0.3307 | 0.0569 | 0.2572 |
| 2 | 0.1048 | 0.2883 | 0.0529 | 0.2362 |

Epoch-1 independent reload reproduced `0.1188/0.3309/0.0569/0.2573`.
Full PAA ranking reduced overall top-1500 proposal recall at IoU50/75 from
the two-epoch local baseline `0.8555/0.3250` to `0.8095/0.3099`. Micro
top-1500 IoU50 fell from `0.7141` to `0.5272`, while large IoU75 rose from
`0.4949` to `0.6065`.

Hard size thresholds at normalized decoded sizes `0.025` and `0.05` were
also rejected. Mixing presence-only and fused scores with a hard boundary
made score scales discontinuous and displaced proposals between size bands.

A continuous geometric blend was then introduced:

```text
score = presence^(1 - w/2) * predicted_iou^(w/2)
```

`w=0` isolates the auxiliary-training effect; `w=1` is the PAA score.
Audits at `w=0`, `0.25`, and `0.5` showed that softer fusion recovers much
of the proposal ranking, but `w=0` still trails the baseline. Therefore the
auxiliary gradient through the shared tower is itself harmful.

## Detached-Tower EMA8 Run

The final bounded variant uses a separate `3x3` IoU tower fed by detached
backbone features. IoU loss updates only that tower and its predictor; it
cannot update the detector backbone, shared RPN tower, objectness head, or
bbox head. Proposal fusion uses `w=0.5`.

This variant received the full local eight-epoch EMA schedule, seed 42, and
the same fixed-scale iterative-CBL configuration as the trainable leader.
The best checkpoint is exact EMA epoch 4.

| Epoch | Candidate AP | Candidate AP75 | Baseline AP | Baseline AP75 |
|---:|---:|---:|---:|---:|
| 1 | 0.0376 | 0.0097 | 0.0355 | 0.0107 |
| 2 | 0.1116 | 0.0484 | 0.1120 | 0.0489 |
| 3 | 0.1353 | 0.0686 | 0.1370 | 0.0693 |
| 4 | **0.1459** | **0.0757** | 0.1458 | 0.0733 |
| 5 | 0.1458 | 0.0730 | **0.1486** | **0.0764** |
| 6 | 0.1407 | 0.0676 | 0.1452 | 0.0728 |
| 7 | 0.1351 | 0.0649 | 0.1399 | 0.0671 |
| 8 | 0.1292 | 0.0611 | 0.1341 | 0.0620 |

Independent reload of candidate epoch 4 gives:

| Variant | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Iterative-CBL EMA epoch 5 leader | **0.1486** | **0.4030** | **0.0764** | **0.2949** |
| Detached RPN IoU EMA epoch 4 | 0.1460 | 0.3866 | 0.0758 | 0.2923 |

The candidate nearly ties strict localization but does not create a new
frontier. It loses `0.0026` AP, `0.0164` AP50, `0.0006` AP75, and `0.0026`
AR100.

## Proposal Diagnosis

Full-validation proposal recall compares each method's selected EMA
checkpoint:

| Scope | Metric | Leader | Detached IoU | Delta |
|---|---|---:|---:|---:|
| Overall | top-100 IoU50 | 0.6045 | 0.5876 | -0.0169 |
| Overall | top-100 IoU75 | 0.1935 | 0.1869 | -0.0066 |
| Overall | top-1500 IoU50 | 0.8666 | 0.8616 | -0.0050 |
| Overall | top-1500 IoU75 | 0.3193 | 0.2976 | -0.0218 |
| Micro | top-1500 IoU75 | 0.1552 | 0.1339 | -0.0213 |
| Tiny | top-100 IoU50 | 0.6209 | **0.6552** | **+0.0343** |
| Tiny | top-100 IoU75 | 0.1926 | **0.1990** | **+0.0064** |
| Small | top-1500 IoU75 | 0.3463 | 0.3122 | -0.0341 |
| Large | top-100 IoU75 | 0.3521 | **0.3945** | **+0.0424** |

The tempered detached branch improves early tiny and large ranking, but it
still favors easier/larger localization at the expense of micro and small
coverage. That scale tradeoff explains why AP75 approaches the leader while
AP, AP50, and AR stay lower.

## Decision

Reject the current RPN IoU-quality formulation for promotion. Do not launch
Kaggle, use the locked test, or sweep fusion/loss weights. The full eight-epoch
run is stable and much better than the shared-tower version, but it does not
beat the existing trainable checkpoint and its proposal audit exposes a
consistent micro/small regression.

Keep the implementation default-off as a documented ablation. The trainable
leader remains iterative CBL EMA epoch 5. The strongest offline validation
pipeline remains scale TTA plus size-aware pair calibration.

Implementation branch: `cbl-rpn-iou-quality-20260731`, commit `113ce29`.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__rpniouw0.5f0.5dt__la_loss__seed42__cbl_iterative_rpn_iou_detached_ema8/metrics.csv`
- `runs/cbl_iterative_rpn_iou_detached_ema8_best_ap75_valid_reload.json`
- `runs/rpn_proposal_recall_cbl_iterative_rpn_iou_detached_ema8_ep4_valid.json`
- `runs/rpn_proposal_recall_cbl_iterative_train_ema8_ep5_valid.json`
- `runs/cbl_iterative_rpn_iou_quality_local_gate_best_ap75_valid_reload.json`
- `runs/rpn_proposal_recall_cbl_iterative_rpn_iou_quality_ep1_valid.json`

## Related Pages

- [[CBL Learned RPN Cascade Local Gate - 2026-07-31]]
- [[CBL RPN Quality Objectness Local Gate - 2026-07-31]]
- [[CBL RPN Proposal Coverage Audit - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
