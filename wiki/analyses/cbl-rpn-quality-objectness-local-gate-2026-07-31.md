---
title: CBL RPN Quality Objectness Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [neurips:2020-gfl, cvpr:2021-vfnet, common/model.py, runs/cbl_iterative_rpn_qfl_clean_best_ap75_valid_reload.json, runs/cbl_iterative_rpn_qfl_microguard_best_ap75_valid_reload.json, runs/rpn_proposal_recall_cbl_iterative_local_ep2_valid.json, runs/rpn_proposal_recall_cbl_iterative_rpn_qfl_ep2_valid.json, runs/rpn_proposal_recall_cbl_iterative_rpn_qfl_microguard_ep2_valid.json]
tags: [cbl, rpn, quality-focal-loss, objectness, proposal-ranking, negative-result]
---

# CBL RPN Quality Objectness Local Gate - 2026-07-31

## Question

Can RPN objectness learn localization quality so that high-IoU proposals rank
earlier, without adding a proposal cascade or quality branch?

## Motivation

The baseline audit found a large ranking gap: overall proposal IoU75 recall
increases from `0.1725` at top 100 to `0.3250` at top 1500. Generalized Focal
Loss and VarifocalNet train a joint foreground/localization-quality score to
improve candidate ranking.

Primary references:

- [Generalized Focal Loss, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html)
- [VarifocalNet, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html)

This experiment is an RPN adaptation, not a reproduction of either full
detector.

## Method

For each sampled positive RPN anchor, the implementation decodes the predicted
and target deltas relative to the same unit anchor. Their aligned IoU becomes a
detached continuous objectness target. Sampled negatives retain target `0`.
Binary Quality Focal Loss with `beta=2` replaces only RPN objectness BCE.
RPN box regression, RoI losses, proposal filtering, and inference are
unchanged.

The CUDA test verifies exact-IoU targets, finite gradients to `cls_logits`,
empty-GT handling, and compatibility with trainable iterative CBL.

The first local run showed positive epoch-1 metrics but entered the known
Windows post-validation paging slowdown. It was stopped without using its
partial epoch 2. Both reported gates below were rerun continuously with
`TOD_EMPTY_CACHE_EVERY=1`, raw weights, no workers, and the same two-epoch
budget as the local CBL leader.

## Detector Result

| Variant / selected epoch | AP | AP50 | AP75 | AR100 | Weighted class-aware AP | Micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| Local CBL leader / epoch 2 | **0.1269** | **0.3612** | **0.0572** | **0.2758** | **0.5084** | **0.3578** |
| RPN QFL / epoch 2 reload | 0.1173 | 0.3356 | 0.0531 | 0.2732 | 0.4787 | 0.2866 |
| RPN QFL micro guard / epoch 1 reload | 0.1192 | 0.3339 | 0.0557 | 0.2669 | 0.4735 | 0.3338 |

Plain QFL underperforms the leader on every detector metric. Its micro
class-aware AP falls sharply.

## Proposal Diagnosis

Full-validation top-1500 proposal recall at the same two-epoch checkpoint
budget:

| Variant | Overall IoU50/75 | Micro IoU50/75 | Tiny IoU50/75 | Small IoU50/75 | Large IoU50/75 |
|---|---:|---:|---:|---:|---:|
| Baseline | `0.8555/0.3250` | `0.7141/0.1541` | `0.8717/0.3151` | `0.9220/0.3950` | `0.9134/0.4949` |
| RPN QFL | `0.8544/0.3475` | `0.6653/0.1386` | `0.8882/0.3473` | `0.9156/0.3845` | `0.9641/0.6350` |
| Micro guard | `0.8197/0.3069` | `0.7514/0.1718` | `0.8064/0.2808` | `0.8413/0.3317` | `0.9263/0.5576` |

Plain QFL succeeds at its intended ranking objective: overall top-100 IoU75
recall rises `0.1725 -> 0.2049`, and top-1500 rises
`0.3250 -> 0.3475`. It strongly improves large and tiny IoU75, but suppresses
micro proposals.

One pre-registered scale guard preserves binary target `1` only for matched GT
with normalized sqrt-area below `8/512=0.015625`. It restores and exceeds
baseline micro recall, but those scores displace too many tiny and small
proposals. Overall proposal recall and detector AP remain negative.

## Decision

Reject single-logit RPN quality objectness for promotion. Run no beta,
threshold, floor, or blend sweep; no Kaggle and no locked test.

The experiment establishes two useful constraints:

1. localization-aware RPN ranking can materially increase strict proposal
   recall;
2. foreground presence and localization quality compete across object scales
   when represented by one score.

The next proposal method should separate the two signals or train a genuine
coarse-to-fine RPN with detached refined anchors, re-matching, and residual
regression. A dual-score or cascade design must demonstrate per-scale
proposal gains before receiving a full training budget.

## Artifacts

- `runs/cbl_iterative_rpn_qfl_clean_best_ap75_valid_reload.json`
- `runs/cbl_iterative_rpn_qfl_microguard_best_ap75_valid_reload.json`
- `runs/rpn_proposal_recall_cbl_iterative_local_ep2_valid.json`
- `runs/rpn_proposal_recall_cbl_iterative_rpn_qfl_ep2_valid.json`
- `runs/rpn_proposal_recall_cbl_iterative_rpn_qfl_microguard_ep1_valid.json`
- `runs/rpn_proposal_recall_cbl_iterative_rpn_qfl_microguard_ep2_valid.json`

## Related Pages

- [[CBL RPN Proposal Coverage Audit - 2026-07-31]]
- [[CBL Iterative RPN Refinement Gate - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
