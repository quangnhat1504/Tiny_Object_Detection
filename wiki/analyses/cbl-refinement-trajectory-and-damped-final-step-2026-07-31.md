---
title: CBL Refinement Trajectory and Damped Final Step - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources:
  - https://openaccess.thecvf.com/content_ECCV_2018/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.html
  - https://openaccess.thecvf.com/content_cvpr_2016/html/Najibi_G-CNN_An_Iterative_CVPR_2016_paper.html
tags: [cbl, localization, iterative-refinement, trajectory, validation]
---

# CBL Refinement Trajectory and Damped Final Step - 2026-07-31

## Question

Do different detections prefer different refinement passes, and can an
observable pass selector or trajectory fusion improve the three-pass CBL
leader without changing weights?

This follows the negative unrolled-training gate. The successful checkpoint
is kept fixed, and all candidates operate only on its inference trajectory.

## Cross-Fit Trajectory Diagnosis

`scripts/analyze_refinement_trajectory.py` captures the pre-refinement boxes
and passes 1-3 once, then evaluates fixed passes, deterministic fusion, a
cross-fit selector, and a GT-only oracle. The selector is trained on even tiles
and applied to odd tiles, and vice versa. Its bins use predicted size and
first-step self-IoU, so no detection is selected using its own GT.

The cache harness matches the independent evaluator within `0.0003` on fixed
pass and scale-gated controls. On the `23,778` detections eligible for
refinement, the GT oracle chooses:

| Best pass | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| Count | 10,137 | 1,636 | 2,087 | 9,918 |
| Rate | 42.63% | 6.88% | 8.78% | 41.71% |

The oracle raises AP/AP75/AR100 to `0.1662/0.0970/0.3233`, so the trajectory
contains substantial complementary localization. That upper bound is not
deployable because it uses validation GT.

Predicted size alone selects pass 3 in every bin on both folds. Adding
self-IoU changes some bins but reaches only AP/AP50/AP75/AR100 =
`0.1503/0.4069/0.0778/0.2937`. This confirms the earlier finding that movement
stability is not a strong enough localization-quality signal. Median fusion
also fails to promote.

## Damped Final Update

The useful fixed rule is not pass selection but damping the last update. Passes
1 and 2 use the full predicted delta; pass 3 moves halfway from `B2` toward
`B3`. A bounded cache sweep of final fractions `0.25/0.50/0.75` selected
`0.50`. The core implementation exposes
`cbl_refine_last_step_blend`; `None` inherits the common blend for backward
compatibility.

Independent full-validation evaluation gives:

| Profile | AP | AP50 | AP75 | AR100 | class-aware scale AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| Three passes, full final update | 0.1501 | 0.4074 | 0.0774 | 0.2934 | **0.5461** | 0.3818 |
| Three passes, 12 px gate | 0.1504 | **0.4081** | 0.0772 | 0.2946 | 0.5449 | **0.3831** |
| **Three passes, final blend 0.50** | **0.1505** | 0.4077 | **0.0781** | 0.2943 | 0.5454 | 0.3797 |
| **12 px gate, final blend 0.50** | **0.1505** | 0.4072 | 0.0779 | **0.2953** | 0.5450 | **0.3831** |

The ungated damped profile is the new standard COCO AP/AP75 leader. The
scale-gated damped profile ties its AP and is the new AR100 leader. The
undamped profile still has the best class-aware scale AP, so these are
deployment profiles rather than a claim that every metric improves.

## Center-Size Motion Decomposition

A follow-up decomposed the final `B2 -> B3` update into center translation and
width/height change. The bounded cache audit also tested direction-cosine and
update-growth gates. Motion gates failed to beat global damping, which means
trajectory reversal is not a reliable per-box quality signal.

Three center/size profiles were independently evaluated:

| Final center/size blend | Extra-pass gate | AP | AP50 | AP75 | AR100 |
|---|---|---:|---:|---:|---:|
| **0.25 / 0.50** | None | 0.1502 | 0.4070 | **0.0787** | 0.2941 |
| 0.50 / 0.25 | None | **0.1505** | **0.4080** | 0.0779 | 0.2946 |
| 0.25 / 0.50 | 12 px | 0.1501 | 0.4067 | 0.0778 | 0.2947 |

Center `0.25`, size `0.50` is the new maximum-AP75 profile, improving the
scalar-damped leader from `0.0781` to `0.0787`. It does not replace the
standard-AP profile because AP falls from `0.1505` to `0.1502`.

Center `0.50`, size `0.25` does not extend the Pareto frontier: it ties AP but
loses AP75 to scalar damping and AR to scale-gated scalar damping. Applying the
12 px gate to the strict profile also regresses. Stop the center/size sweep.

## AP75 Error Audit

The score-0.05/top-100 audit compares the ungated damped profile with the full
three-pass control:

- audit AP75: `0.0788 -> 0.0796`
- greedy TP75: `1,728 -> 1,724`
- micro TP75: `70 -> 72`
- tiny TP75: `565 -> 572`
- small TP75: `978 -> 963`
- large TP75: `115 -> 117`
- background predictions: `70,722 -> 70,591`

AP75 improves despite four fewer greedy TP75 detections. The final damping
changes box geometry and NMS ordering enough to improve the precision-recall
curve, while shifting strict hits from small objects toward micro, tiny, and
large objects. It is a real but narrow inference gain.

The center/size strict profile raises audit AP75 again from `0.0796` to
`0.0798`, raises greedy TP75 from `1,724` to `1,727`, and lowers background
predictions from `70,591` to `70,536`. Scale TP75 changes from the scalar
damped profile as micro `72 -> 71`, tiny `572 -> 575`, small `963 -> 962`,
and large `117 -> 119`.

## Decision

- Use three ungated passes with final blend `0.50` for maximum standard COCO
  AP and the best overall AP/AP75 balance.
- Use the 12 px extra-pass gate with final blend `0.50` for maximum AR100 and
  the best AP/recall balance.
- Use final center/size blends `0.25/0.50` without a scale gate for maximum
  strict AP75.
- Retain full-update three-pass inference when class-aware scale AP is the
  primary metric.
- Reject learned size-only and size-plus-stability pass selectors, median
  fusion, motion gates, and GT-oracle selection.
- Do not run Kaggle or reopen the locked test: weights are unchanged and the
  CBL family has already consumed its frozen test gate.

## Artifacts

- `runs/cbl_refinement_trajectory_ema8_ep5_valid_v2.json`
- `runs/cbl_iterative_train_ema8_step3_lastblend050_valid.json`
- `runs/cbl_iterative_train_ema8_step3_scale12_lastblend050_valid.json`
- `runs/ap75_analysis_cbl_iterative_train_ema8_ep5_step3_lastblend050_valid/summary.json`
- `runs/cbl_refinement_motion_ema8_ep5_valid.json`
- `runs/cbl_iterative_train_ema8_step3_center025_size050_valid.json`
- `runs/cbl_iterative_train_ema8_step3_center050_size025_valid.json`
- `runs/cbl_iterative_train_ema8_step3_scale12_center025_size050_valid.json`
- `runs/ap75_analysis_cbl_iterative_train_ema8_ep5_step3_center025_size050_valid/summary.json`

Implementation transport:

- branch `cbl-iterative-depth-20260731`
- commits `bfc409a`, `de5d4b8`

## Related Pages

- [[CBL Refinement Consistency and Depth Gate - 2026-07-31]]
- [[CBL Horizontal-Flip TTA Local Gate - 2026-07-31]]
- [[Unrolled Iterative CBL Training Local Gate - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[Wiki Overview]]
- [[Wiki Log]]
