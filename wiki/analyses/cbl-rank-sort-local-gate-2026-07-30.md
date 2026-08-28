---
title: CBL Rank and Sort Local Gate - 2026-07-30
type: analysis
created: 2026-07-30
updated: 2026-07-30
sources: [iccv:2021-rank-sort-loss, github:kemaloksuz-ranksortloss, common/model.py, runs/sa_alw_full__cbl__rsd0.5__la_loss__seed42__cbl_ranksort_local_gate/metrics.csv]
tags: [cbl, rank-sort, classification, localization-quality, ap75, negative-result]
---

# CBL Rank and Sort Local Gate - 2026-07-30

## Question

Can a two-stage ranking objective improve CBL score-localization alignment
after the dense-detector QFL transfer failed?

## Research Basis

Rank & Sort Loss (ICCV 2021) jointly ranks positives above negatives and sorts
positive scores by localization IoU. Its official implementation reports about
three box AP improvement for Faster R-CNN, making it a better two-stage
candidate than another QFL parameter sweep.

The official RS-R-CNN recipe also removes sampling, replaces localization with
GIoU, adds self-balancing, modifies RPN training, and tunes learning rate. The
first project gate deliberately isolates only sampled RoI classification:

- foreground sigmoid logits, excluding the background logit;
- one positive target per sampled foreground RoI, equal to detached paired CBL
  IoU;
- other classes and sampled background RoIs use target zero;
- official identity-update ranking and sorting gradients with `delta=0.5`;
- CBL, RPN, sampler, data, augmentation, optimizer, and inference budget stay
  unchanged.

This is a bounded adaptation, not a reproduction of the full RS-R-CNN recipe.

## Technical Verification

- device-agnostic port matches the official comparison and identity-update
  equations;
- high-scoring relevant negatives receive positive gradients;
- positives receive negative gradients and irrelevant negatives receive zero;
- reversing positive score order relative to IoU creates a non-zero sorting
  error;
- CUDA AMP forward/backward, inference, checkpoint reload, standard CBL, and
  QFL regression smokes passed.

## Validation Gate

Both candidates use seed 42, raw weights, no workers, CBL, copy-paste, and
tiny-tile oversampling.

| Model | Epoch | AP | AP50 | AP75 | AR100 | weighted class-aware AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| CBL standard | 1 | **0.1145** | **0.3334** | **0.0454** | **0.2692** | - | - |
| CBL + sampled RS | 1 | 0.0953 | 0.2909 | 0.0327 | 0.2542 | 0.4224 | 0.1992 |
| CBL standard | 2 | **0.1200** | **0.3523** | **0.0471** | **0.2759** | **0.4938** | **0.3515** |
| CBL + sampled RS | 2 | 0.0823 | 0.2509 | 0.0293 | 0.2570 | 0.3932 | 0.2388 |

Independent reload reproduced the best epoch-1 checkpoint at
AP/AP50/AP75/AR100=`0.0953/0.2910/0.0327/0.2542`.

Epoch 2 slowed after epoch-1 validation, from about two batches per second to
8.8 seconds per batch. A clean restart from `last.pt` restored normal
throughput and completed epoch 2. This recurring process-lifecycle issue does
not explain the metric regression.

## Score-IoU Diagnosis

Raw tile predictions were cached at score threshold `0.001` and compared
directly with the standard local CBL checkpoint:

| Checkpoint | Pearson | Spearman | mean score | IoU50 GT coverage | IoU75 GT coverage |
|---|---:|---:|---:|---:|---:|
| CBL standard epoch 2 | **0.6181** | **0.5225** | 0.0802 | **0.7312** | **0.2115** |
| CBL + sampled RS epoch 1 | 0.4977 | 0.4786 | 0.4483 | 0.7179 | 0.1864 |

The count of predictions with score at least `0.2` but same-class IoU below
`0.5` increased from `33,676` to `343,905`. RS therefore did not trade AP for
better score-IoU ordering; it made ranking and calibration worse in this
sampled CBL setting.

## Decision

Negative local gate. Do not launch sampled CBL+RS on Kaggle, do not open the
locked test set, and do not sweep `delta`. The full official recipe changes too
many components to infer that a `delta` tweak would repair this failure.

Revisit Rank & Sort only as a separately budgeted full RS-R-CNN reproduction
with unsampled RoIs, RPN changes, GIoU weighting, self-balancing, and a tuned
schedule. It is not the next low-risk route for the current CBL leader.

## Artifacts

- `runs/sa_alw_full__cbl__rsd0.5__la_loss__seed42__cbl_ranksort_local_gate/metrics.csv`
- `runs/cbl_ranksort_local_gate_best_ap75_valid_reload.json`
- `runs/quality_diagnosis_valid_cbl_ranksort_ep1.json`
- `runs/quality_diagnosis_valid_cbl_local_ep2.json`
- `scripts/test_rank_sort_cbl.py`

## Related Pages

- [[CBL EMA Recovery Audit - 2026-07-30]]
- [[CBL Quality Focal Loss Local Gate - 2026-07-30]]
- [[Confidence-Driven Localization Local Gate - 2026-07-30]]
