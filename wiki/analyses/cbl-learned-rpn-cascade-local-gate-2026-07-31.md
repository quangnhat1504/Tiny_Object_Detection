---
title: CBL Learned RPN Cascade Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [iccv:2023-cfinet, common/model.py, runs/cbl_iterative_rpn_cascade_local_gate_best_ap75_valid_reload.json, runs/rpn_proposal_recall_cbl_iterative_local_ep2_valid.json, runs/rpn_proposal_recall_cbl_iterative_rpn_cascade_local_ep2_valid.json]
tags: [cbl, rpn, cascade, coarse-to-fine, proposal-recall, negative-result]
---

# CBL Learned RPN Cascade Local Gate - 2026-07-31

## Question

Can a learned regression-only RPN stage refine anchors before a separately
trained objectness/residual-regression stage, improving strict proposal recall
and detector AP75 over the trainable iterative-CBL leader?

## Method

The implementation is a bounded CFINet-inspired cascade:

1. a dilation-3, regression-only RPN head predicts coarse anchor deltas;
2. the deltas are decoded and detached to form refined anchors;
3. refined anchors are re-matched with the existing SA-ALW hierarchical
   assignment;
4. the standard RPN head predicts objectness and residual deltas relative to
   the refined anchors.

Stage 1 uses Smooth-L1 with weight `1.0`; stage 2 keeps the existing RPN loss.
The Faster R-CNN backbone, RoI losses, trainable one-pass CBL refinement, seed,
resize, data, and two-epoch budget are unchanged.

This is not a full reproduction of
[CFINet](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html).
The official Cascade RPN assumes one anchor per location and uses adaptive
feature convolution, learned offsets, and bridged features. This project uses
nine anchors per location, so the bounded gate tests detached residual
re-matching without those components.

Focused CUDA tests verified finite losses, gradients to both RPN stages,
no stage-2 gradient through detached refined anchors, deterministic checkpoint
reload, and compatibility with iterative CBL. Existing fixed-delta, QFL, SNIP,
and iterative-CBL tests also passed.

## Detector Result

Both runs use raw weights, seed 42, no workers, cache cleanup each batch, and
the same two-epoch local budget.

| Variant / selected epoch | AP | AP50 | AP75 | AR100 | Weighted class-aware AP | Micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| Local CBL leader / epoch 2 reload | **0.1269** | **0.3612** | **0.0572** | **0.2758** | **0.5084** | **0.3578** |
| Learned RPN cascade / epoch 2 reload | 0.1094 | 0.2914 | 0.0552 | 0.2486 | 0.3981 | 0.1813 |

The cascade loses `0.0175` AP and `0.0272` AR100. AP75 is closer but still
loses `0.0020`. Epoch time rises from about `634` to `803` seconds, a `26.6%`
increase. Independent reload reproduces the selected epoch-2 result, so the
gap is not checkpoint metadata drift.

## Proposal Diagnosis

Full-validation proposal recall at the same epoch-2 budget:

| Scope | Top-100 IoU50 | Top-100 IoU75 | Top-1500 IoU50 | Top-1500 IoU75 |
|---|---:|---:|---:|---:|
| Baseline overall | **0.5764** | **0.1725** | **0.8555** | **0.3250** |
| Cascade overall | 0.5228 | 0.1638 | 0.8313 | 0.3223 |
| Baseline micro | 0.4395 | 0.0919 | 0.7141 | 0.1541 |
| Cascade micro | **0.4551** | **0.1121** | **0.7488** | **0.1858** |
| Baseline tiny | **0.6074** | **0.1658** | **0.8717** | 0.3151 |
| Cascade tiny | 0.5452 | 0.1551 | 0.8474 | **0.3376** |
| Baseline small | **0.5952** | **0.1945** | **0.9220** | **0.3950** |
| Cascade small | 0.4994 | 0.1644 | 0.8599 | 0.3309 |

The learned coarse stage materially improves micro strict-localization
coverage, but the second-stage score ranking displaces tiny/small proposals.
Overall top-100 IoU50 falls by `0.0535`, and small top-1500 IoU75 falls by
`0.0641`. The RoI detector then converts neither the micro proposal gain nor
the nearly tied overall IoU75 coverage into AP; micro class-aware AP roughly
halves.

## Decision

Reject this simplified RPN cascade. Do not sweep stage-1 weight, dilation,
depth, or training duration; no Kaggle and no locked test. The local gate loses
all primary detector metrics, adds substantial runtime, and lacks the adaptive
feature alignment that defines the official method.

The combined RPN evidence now favors a separated two-signal test: retain
binary foreground presence for proposal coverage and learn localization
quality in an auxiliary positive-only branch. Any fused ranking must first
preserve per-scale top-N recall before receiving a full detector budget.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__rpncasw1__la_loss__seed42__cbl_iterative_rpn_cascade_local_gate/metrics.csv`
- `runs/cbl_iterative_rpn_cascade_local_gate_best_ap75_valid_reload.json`
- `runs/rpn_proposal_recall_cbl_iterative_local_ep2_valid.json`
- `runs/rpn_proposal_recall_cbl_iterative_rpn_cascade_local_ep2_valid.json`

## Related Pages

- [[CBL RPN Proposal Coverage Audit - 2026-07-31]]
- [[CBL Iterative RPN Refinement Gate - 2026-07-31]]
- [[CBL RPN Quality Objectness Local Gate - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
