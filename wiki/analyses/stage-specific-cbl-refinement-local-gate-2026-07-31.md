---
title: Stage-Specific CBL Refinement Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [cvpr:2018-cascade-rcnn, common/model.py, runs/cbl_stage2_local_gate_best_ap75_valid_reload.json]
tags: [cbl, cascade, iterative-refinement, localization, negative-result]
---

# Stage-Specific CBL Refinement Local Gate - 2026-07-31

## Question

Does a separate second-pass CBL regression head improve over the positive
shared-head trainable refinement result?

## Method

- Clone the first-pass `TwoMLPHead` and CBL predictor at initialization.
- Train the clone only on detached first-pass positive proposals.
- Freeze and ignore the cloned classifier.
- Preserve the first-pass class labels and scores at inference.
- Use the stage-specific regressor for exactly one refinement pass.

This isolates regression specialization without adding second-stage
classification or changing proposal matching.

## Verification

- Shared-head regression smoke remains exact.
- Stage-specific CUDA loss is finite (`1.5385`).
- Gradients reach the cloned `fc7` and CBL distribution layer; the unused
  cloned classifier receives no gradient.
- Inference and checkpoint reload are exact.
- Batch size 4 peak memory is `3.716 GiB` allocated and `3.811 GiB` reserved.

## Two-Epoch Result

| Method | AP | AP50 | AP75 | AR100 | weighted class-aware AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| Standard CBL | 0.1199 | 0.3523 | 0.0467 | 0.2759 | 0.4938 | 0.3515 |
| Shared-head trainable refinement | **0.1269** | **0.3612** | **0.0572** | **0.2758** | **0.5084** | **0.3578** |
| Stage-specific regression reload | 0.1218 | 0.3420 | 0.0547 | 0.2729 | 0.4796 | 0.2798 |

The separate regressor retains an AP75 gain over standard CBL, but loses AP,
AP50, AR, and class-aware accuracy relative to shared-head refinement.

## Decision

Negative promotion gate. Do not launch this regression-only variant on
Kaggle and do not tune its loss weight blindly.

The result does not reject a full Cascade R-CNN stage. It rejects only
parameter separation without second-stage re-matching/classification. The
next cascade experiment must address background predictions explicitly with
refined-proposal sampling and a stage-2 classification target; otherwise the
extra head adds capacity without correcting the observed error source.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__irh2__la_loss__seed42__cbl_stage2_local_gate/metrics.csv`
- `runs/cbl_stage2_local_gate_best_ap75_valid_reload.json`
- branch `cbl-stage2-refine-20260731`

## Related Pages

- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[Iterative CBL Refinement Gate - 2026-07-31]]
- [[Double-Head CBL Local Gate - 2026-07-31]]
- [[Wiki Log]]
