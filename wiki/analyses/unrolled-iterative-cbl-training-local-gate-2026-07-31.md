---
title: Unrolled Iterative CBL Training Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources:
  - https://openaccess.thecvf.com/content_cvpr_2016/html/Najibi_G-CNN_An_Iterative_CVPR_2016_paper.html
  - https://openaccess.thecvf.com/content_cvpr_2018/html/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.html
tags: [cbl, localization, iterative-refinement, training, negative-result]
---

# Unrolled Iterative CBL Training Local Gate - 2026-07-31

## Question

Can the shared CBL box head improve its inference-time fixed point by training
on the same three-step refinement trajectory used at inference?

The prior trainable leader supervises only `B0 -> B1`, although its best
inference profiles reapply the head through `B2` and `B3`. This ablation
unrolls the detached trajectory during training and applies the same shared
head at all three passes. It tests whether train/inference alignment is enough
to improve localization without adding a cascade head.

## Implementation And Technical Gate

`cbl_refine_train_steps` controls the number of detached training passes and
defaults to one for backward compatibility. For three-step training:

1. Each pass decodes the current CBL deltas and detaches the refined proposals.
2. The same RoI box head and CBL predictor are reused at the next pass.
3. Per-pass CBL losses are weighted by active proposal count and normalized so
   the total auxiliary weight remains `0.50`.
4. The scale-gated variant applies later training passes only to boxes whose
   normalized predicted size is at least `0.0234375`, equivalent to 12 px in a
   512 px tile.

CUDA forward/backward/reload passed. A batch-size-4 gate used `3.6639/3.7539`
GiB allocated/reserved memory for both variants. The scale-gated example used
active counts `[141, 7, 7]`; the ungated control used `[141, 141, 141]`.
All pass losses and CBL gradients were finite.

## Two-Epoch Local Gate

Both runs use raw weights, the same data recipe and seed 42, loss weight
`0.50`, three train steps, three inference steps, and refinement score
threshold `0.30`.

| Training profile | Epoch | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|---:|
| Scale-gated at 12 px | 1 | 0.1188 | 0.3385 | 0.0543 | 0.2632 |
| **Scale-gated at 12 px** | **2** | **0.1249** | **0.3591** | **0.0559** | **0.2685** |
| **Ungated** | **1** | **0.1206** | **0.3453** | **0.0536** | **0.2654** |
| Ungated | 2 | 0.1160 | 0.3378 | 0.0517 | 0.2603 |
| One-pass-trained local leader | 2 | **0.1269** | **0.3612** | **0.0572** | **0.2758** |

Independent `best_ap75.pt` reloads exactly reproduce scale-gated epoch 2 and
ungated epoch 1. The scale-gated checkpoint reaches class-aware scale/micro AP
`0.5063/0.2951`; the ungated checkpoint reaches `0.4957/0.3094`. The
one-pass-trained leader remains higher at `0.5084/0.3578`.

The scale-gated checkpoint still benefits from inference depth:

| Inference profile | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| 1 step | 0.1214 | 0.3500 | 0.0533 | **0.2691** |
| 2 steps, 12 px gate | 0.1237 | 0.3569 | 0.0542 | 0.2688 |
| 3 steps, 12 px gate | **0.1249** | **0.3591** | **0.0559** | 0.2685 |

This shows that the inference operation remains useful, but the three-pass
training distribution moves the shared head below the one-pass-trained model.

## Interpretation

The sparse scale gate prevents the epoch-2 collapse seen in the ungated run,
but it cannot recover the leader. Ungated deep supervision exposes the same
parameters to progressively shifted proposal distributions at every batch.
The head improves at epoch 1 and then degrades across AP, AP75, and AR at
epoch 2. The likely failure is a shared-head fixed-point conflict: optimizing
later refined boxes changes the regressor used for the base proposals, while a
true cascade avoids this conflict with stage-specific heads and matching.

This result narrows the successful recipe:

- train the CBL head with one detached refinement loss;
- repeat that trained head only at inference;
- gate extra inference passes by predicted size when balanced AP/AR matters.

## Decision

- Reject both three-step training variants.
- Do not run them on Kaggle or consume another locked-test gate.
- Do not sweep unroll loss weights, train depth, or more size thresholds.
- Retain the one-pass-trained EMA epoch-5 checkpoint and its positive
  inference-only three-step profiles.
- Study the existing per-pass trajectory before introducing another training
  head: measure whether pass selection or trajectory fusion can exploit
  complementary localization without changing weights.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5it3ir3s0.3m0.0234375__la_loss__seed42__cbl_iterative_train3_scale12_local_gate/metrics.csv`
- `runs/cbl_iterative_train3_scale12_local_gate_best_ap75_valid_reload.json`
- `runs/cbl_iterative_train3_scale12_local_gate_best_ap75_step1_valid.json`
- `runs/cbl_iterative_train3_scale12_local_gate_best_ap75_step2_scale12_valid.json`
- `runs/sa_alw_full__cbl__irtw0.5it3ir3s0.3__la_loss__seed42__cbl_iterative_train3_ungated_local_gate/metrics.csv`
- `runs/cbl_iterative_train3_ungated_local_gate_best_ap75_valid_reload.json`

Implementation archive:

- branch `cbl-unrolled-train-20260731`
- commit `deea9a7`

## Related Pages

- [[CBL Refinement Consistency and Depth Gate - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[CBL Cascade Stage-2 Local Gate - 2026-07-31]]
- [[Wiki Overview]]
- [[Wiki Log]]
