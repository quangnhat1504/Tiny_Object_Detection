---
title: CBL RPN Proposal Coverage Audit - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [iccv:2023-cfinet, github:shaunyuan22-cfinet, scripts/audit_rpn_proposal_recall.py, runs/rpn_proposal_recall_cbl_iterative_train_ema8_ep5_valid.json]
tags: [cbl, rpn, proposal-recall, cfinet, tiny-object-detection, diagnosis]
---

# CBL RPN Proposal Coverage Audit - 2026-07-31

## Question

Does the current EMA epoch-5 leader supply enough high-IoU RPN proposals for
the RoI CBL head, especially for micro objects?

## Motivation

CFINet identifies low prior-to-GT overlap as a small-object bottleneck and uses
a coarse-to-fine RPN: a regression-only stage first refines anchors, then a
second stage classifies and regresses the refined anchors. Its dynamic
assignment lowers the positive-IoU threshold for smaller GT.

Primary references:

- [CFINet, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yuan_Small_Object_Detection_via_Coarse-to-fine_Proposal_Generation_and_Imitation_Learning_ICCV_2023_paper.html)
- [CFINet official implementation](https://github.com/shaunyuan22/CFINet)

The current project already has anchors down to 4 px and SA-ALW hierarchical
dynamic top-k assignment with an expanded-anchor pass. Porting only CFINet's
size-dependent threshold would therefore overlap existing behavior. Proposal
coverage must justify the larger cascade change first.

## Audit Protocol

`scripts/audit_rpn_proposal_recall.py` loads the exact reloadable trainable-CBL
EMA epoch-5 leader and runs only transform, backbone, and RPN on all validation
tiles. It measures whether each transformed GT has any post-NMS RPN proposal
at a requested IoU among the first 100, 300, 1000, or 1500 proposals.

- validation: 131 original images, 1,764 tiles;
- tiles with GT: 787;
- clipped GT instances: 8,274;
- RPN proposals: exactly 1,500 per tile;
- size bands use original tile sqrt-area, while IoU uses transformed
  coordinates shared by GT and proposals.

This is proposal recall, not detector AP and not image-level deduplicated
recall.

## Result

### Overall Recall

| Proposal budget | IoU50 | IoU75 |
|---:|---:|---:|
| Top 100 | 0.6045 | 0.1935 |
| Top 300 | 0.7440 | 0.2490 |
| Top 1000 | 0.8416 | 0.3023 |
| Top 1500 | **0.8666** | **0.3193** |

Top-1500 maximum-IoU quantiles are p10=`0.4418`, p25=`0.6041`,
median=`0.7019`, p75=`0.7693`, and p90=`0.8221`.

### Top-1500 Recall by Size

| Size band | GT | IoU50 | IoU75 |
|---|---:|---:|---:|
| Micro `<8` px | 1,927 | 0.7156 | **0.1552** |
| Tiny `8-16` px | 2,799 | 0.8825 | 0.3305 |
| Small `16-32` px | 2,463 | 0.9363 | 0.3463 |
| Large `>=32` px | 1,085 | 0.9355 | 0.5207 |

Increasing proposal budget from 300 to 1500 improves overall IoU75 recall only
from `0.2490` to `0.3193`. The bottleneck is proposal localization quality,
not only a too-small post-NMS budget. Micro GT is the clearest failure mode:
84.5% has no IoU75 RPN proposal even at the maximum budget.

## Decision

The audit supports a bounded RPN refinement/cascade experiment. It does not
support another anchor-size, proposal-count, assignment-threshold, or
multi-scale sweep.

The next implementation should isolate proposal refinement:

1. stage 1 regresses existing anchors with the current size-aware assignment;
2. refined anchors are detached and re-matched;
3. stage 2 predicts objectness and a residual regression target;
4. a pre-training CUDA audit must show higher IoU75 proposal coverage without
   collapsing IoU50 coverage;
5. only then run the same two-epoch local AP/AP75/AR gate.

This is a CFINet-inspired cascade ablation unless adaptive offset convolution
and the paper's complete dynamic assignment are also ported. Keep that caveat
explicit; do not label a simplified cascade as full CFINet.

## Artifact

- `runs/rpn_proposal_recall_cbl_iterative_train_ema8_ep5_valid.json`

## Related Pages

- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[Confidence-Driven Localization Local Gate - 2026-07-30]]
- [[CBL SNIP-Like Scale-Normalized Training Local Gate - 2026-07-31]]
