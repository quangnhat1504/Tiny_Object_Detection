---
title: CBL Iterative RPN Refinement Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [scripts/audit_rpn_proposal_recall.py, scripts/test_coco_eval_single.py, runs/rpn_proposal_recall_cbl_iterative_train_ema8_ep5_pass2_valid.json, runs/cbl_iterative_train_ema8_rpn_refine1_valid.json, runs/cbl_iterative_train_ema8_rpn_refine1_min16_valid.json]
tags: [cbl, rpn, proposal-refinement, localization, negative-result]
---

# CBL Iterative RPN Refinement Gate - 2026-07-31

## Question

Can a second application of the trained RPN box deltas improve proposal
localization and detector AP without training a separate cascade stage?

## Method

The diagnostic reuses one RPN head evaluation and decodes its fixed deltas
twice. The second decode treats pass-one boxes as refined anchors. Objectness,
pre-NMS selection, and NMS remain unchanged. Training is untouched.

The implementation was first required to reproduce the standard torchvision
pass-one proposals exactly. It passed with maximum absolute difference `0`.
Three passes were rejected on a fixed 64-tile subset because pass two had
higher IoU50 and IoU75 proposal recall.

This is a repeat-delta diagnostic. It is not a trained coarse-to-fine RPN and
must not be labelled CFINet.

## Proposal Result

On all 1,764 validation tiles, the second decode changed top-1500 proposal
recall as follows:

| Size band | Baseline IoU50 | Pass-2 IoU50 | Baseline IoU75 | Pass-2 IoU75 |
|---|---:|---:|---:|---:|
| Overall | 0.8666 | 0.8613 | 0.3193 | **0.3424** |
| Micro `<8` px | 0.7156 | 0.7130 | 0.1552 | **0.1744** |
| Tiny `8-16` px | **0.8825** | 0.8700 | **0.3305** | 0.3073 |
| Small `16-32` px | 0.9363 | 0.9367 | 0.3463 | **0.4166** |
| Large `>=32` px | 0.9355 | 0.9309 | 0.5207 | **0.5631** |

The strict-overlap signal is real but mixed: small and large objects benefit,
while tiny-object recall regresses.

## Detector Gate

The same EMA epoch-5 checkpoint was evaluated end to end.

| Variant | AP | AP50 | AP75 | AR100 | Weighted scale AP | Micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.1486 | **0.4030** | **0.0764** | **0.2949** | 0.5395 | 0.3804 |
| Global pass 2 | 0.1481 | 0.3990 | 0.0753 | 0.2914 | 0.5367 | 0.3593 |
| Pass 2 for predicted size `>=16` px | **0.1492** | 0.4023 | 0.0760 | 0.2940 | **0.5421** | **0.3809** |

The pre-registered size gate uses normalized sqrt-area `16/512=0.03125`.
It recovers the micro regression and produces a small aggregate-AP increase,
but AP75 and AR remain below baseline. No threshold sweep was run.

## Decision

Reject fixed-head iterative RPN decoding as a performance method. Keep it only
as default-off diagnostic infrastructure.

The proposal audit still supports a learned proposal-quality intervention:
top-100/top-300 IoU75 recall is much lower than top-1500 recall, so RPN
objectness should be trained to rank localization quality instead of only
binary foreground. A later full coarse-to-fine RPN remains possible, but it
must train residual regression and re-match refined anchors rather than
reapplying the same deltas.

No Kaggle run and no locked-test evaluation are justified.

## Artifacts

- `runs/rpn_proposal_recall_cbl_iterative_train_ema8_ep5_pass2_valid.json`
- `runs/cbl_iterative_train_ema8_rpn_refine1_valid.json`
- `runs/cbl_iterative_train_ema8_rpn_refine1_min16_valid.json`

## Related Pages

- [[CBL RPN Proposal Coverage Audit - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[CBL Refinement Trajectory and Damped Final Step - 2026-07-31]]
