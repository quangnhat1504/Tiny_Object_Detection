---
title: CBL Horizontal-Flip TTA Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources:
  - wiki/research/deep-research-architecture-and-training.md
  - runs/cbl_flip_tta_ema8_ep5_valid.json
  - runs/cbl_flip_tta_scalar_selected_ema8_ep5_valid.json
  - runs/cbl_flip_tta_strict_selected_ema8_ep5_valid.json
  - runs/cbl_flip_tta_scalar_tiny_gate_valid.json
  - runs/cbl_flip_tta_strict_tiny_gate_valid.json
  - runs/cbl_flip_tta_scalar_strict_profile_ensemble_valid.json
tags: [cbl, localization, tta, flip, fusion, validation]
---

# CBL Horizontal-Flip TTA Local Gate - 2026-07-31

## Question

Can a horizontally flipped view provide complementary localization evidence
for the fixed CBL EMA checkpoint, and can paired fusion convert that evidence
into a reproducible AP/AP75 gain?

Multi-scale TTA was recommended in the project research but had no experiment
artifact. Horizontal flip is the bounded first test because it preserves tiny
object scale and requires only two inference views.

## Method

`scripts/eval_cbl_flip_tta.py` runs each validation tile in its original and
horizontally flipped forms. Flip boxes are mapped back to original
coordinates. The selected fusion:

1. greedily pairs original/flip detections of the same class at IoU `>=0.50`;
2. averages paired box coordinates using their class scores as weights;
3. replaces each paired score with the mean of the two scores;
4. preserves unmatched original detections and ignores unmatched flip-only
   detections;
5. applies class-aware NMS at IoU `0.50`.

This combines localization voting with cross-view score consistency. It does
not use GT, retrain the model, or change checkpoint weights.

A follow-up tiny-object variant keeps the original-view box coordinates for
matched pairs whose original predicted sqrt-area is below a fixed cutoff, while
still averaging the paired scores. The bounded audit tested cutoffs `8`, `12`,
and `16` px. The best setting is `<12` px; it is deterministic and uses only
prediction geometry, not GT.

## Full-Validation Gate

The primary run uses the scalar-damped three-pass profile: passes 1/2 are
full, pass 3 has blend `0.50`, and the refinement score threshold is `0.30`.

| Prediction profile | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Original only | 0.1504 | 0.4073 | 0.0783 | 0.2942 |
| Flip only | 0.1479 | 0.4052 | 0.0725 | 0.2912 |
| **Paired flip fusion, IoU 0.50** | **0.1561** | **0.4227** | **0.0785** | **0.2961** |

The AP gain is `+0.0057` inside the byte-identical TTA evaluator. Weighted
class-aware scale AP reaches `0.5609` and micro class-aware AP reaches
`0.3924`, versus approximately `0.5454/0.3797` for the scalar-damped
single-view profile.

The result is stable across the bounded pair-threshold audit:

| Pair threshold | Coordinate/score fusion | AP | AP75 | AR100 |
|---:|---|---:|---:|---:|
| 0.50 | score-weighted box / mean score | **0.1561** | 0.0785 | 0.2961 |
| 0.50 | equal box / mean score | 0.1558 | 0.0784 | 0.2954 |
| 0.60 | score-weighted box / mean score | 0.1556 | 0.0784 | 0.2958 |
| 0.70 | score-weighted box / mean score | 0.1553 | 0.0785 | 0.2962 |

The gain is therefore not isolated to a single matching threshold or box
weighting formula.

## Strict And Recall Profiles

Applying the same fixed fusion to the center/size strict base
(`0.25/0.50`) gives AP/AP50/AP75/AR100 =
`0.1557/0.4207/0.0788/0.2959`. This is the new maximum-AP75 profile, while
scalar-damped TTA remains the overall AP leader.

The tiny-aware follow-up improves strict AP75 further:

| TTA profile | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Scalar paired fusion | **0.1561** | **0.4227** | 0.0785 | **0.2961** |
| Scalar tiny-keep-box `<12` | 0.1554 | 0.4217 | 0.0791 | 0.2955 |
| Strict paired fusion | 0.1557 | 0.4207 | 0.0788 | 0.2959 |
| **Strict tiny-keep-box `<12`** | 0.1551 | 0.4199 | **0.0795** | 0.2954 |

This creates a clear two-profile choice: scalar paired fusion remains the
overall AP/AP50/AR leader, while strict tiny-keep-box `<12` is the AP75-only
leader.

A follow-up profile-ensemble audit combined the scalar AP leader with the
strict tiny AP75 leader from cache. Paired scalar/strict fusion at IoU `0.50`
or `0.60` reaches AP/AP75/AR100=`0.1560/0.0790/0.2960`, below both relevant
single-profile leaders. Union-NMS `0.60` reaches AR100=`0.2969`, but this is
far below the earlier high-recall union profile (`0.3053`). Do not promote
scalar+strict profile ensembling; it costs extra profile inference without a
new AP, AP75, or AR frontier.

Two fusion variants trade localization precision for recall:

| High-recall profile | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Union + NMS 0.60 | 0.1510 | 0.4087 | 0.0760 | **0.3053** |
| Paired max score + unmatched flip at 0.90 | **0.1533** | **0.4132** | **0.0771** | 0.3034 |

These are optional offline recall profiles, not the default high-accuracy
fusion.

## Robustness And Error Audit

The selected scalar fusion improves both deterministic validation halves:

| Fold | Original AP/AP75/AR | Fused AP/AP75/AR |
|---|---|---|
| Even tiles (882) | 0.1473 / 0.0788 / 0.2888 | **0.1528 / 0.0791 / 0.2919** |
| Odd tiles (882) | 0.1540 / 0.0775 / 0.2997 | **0.1599 / 0.0803 / 0.3003** |

At IoU `0.50`, `183,775` pairs are matched from `339,645` original
detections, a 54.1% match rate. The effect is broad rather than driven by a
small set of detections.

The score-0.05/top-100 AP75 audit changes:

- TP75: `1,710 -> 1,743`
- recall75: `0.2067 -> 0.2107`
- localization FP at IoU 0.50-0.75: `5,609 -> 5,533`
- localization FP at IoU 0.25-0.50: `13,868 -> 13,617`
- background predictions: `70,608 -> 69,099`
- scale TP75: micro `+1`, tiny `-6`, small `+37`, large `+1`

The main gain comes from small-object localization and a large reduction in
background predictions. Tiny TP75 remains the main fusion weakness.

The `<12` px tiny-keep-box rule directly targets that weakness. On the scalar
profile it raises AP75 `0.0785 -> 0.0791`; on the strict profile it raises
AP75 `0.0788 -> 0.0795`. Both deterministic validation halves improve AP75
under the strict profile (`0.0799 -> 0.0805`, `0.0801 -> 0.0808`). The strict
AP75 audit changes TP75 `1,744 -> 1,750`, with tiny TP75 `559 -> 564`, small
TP75 `1,000 -> 1,001`, and large TP75 held at `121`.

## Cost

The paired original+flip inference loop processed 1,764 tiles in about 93
seconds, approximately 19 tiles/s before offline fusion/evaluation. This is
roughly half the throughput of the single-view three-pass profile, as expected
from two model views.

## Decision

- Use scalar-damped paired flip fusion at IoU `0.50` for maximum overall AP,
  AP50, class-aware scale AP, and micro AP.
- Use strict center/size paired flip fusion plus tiny-keep-box `<12` for
  maximum AP75.
- Use union-NMS or unmatched-flip fusion only when AR100 is the primary goal.
- Do not ensemble scalar and strict TTA profiles; the cache audit adds cost
  without a new frontier.
- Keep single-view profiles for latency-sensitive deployment.
- Do not run Kaggle or reopen the locked test: the weights are unchanged and
  the CBL family has already consumed its frozen test gate.
- Do not begin a broad flip-fusion threshold sweep; the gain already repeats
  over 0.50-0.70.

## Artifacts

- `runs/cbl_flip_tta_ema8_ep5_valid.json`
- `runs/cbl_flip_tta_scalar_selected_ema8_ep5_valid.json`
- `runs/cbl_flip_tta_strict_selected_ema8_ep5_valid.json`
- `runs/cbl_flip_tta_scalar_tiny_gate_valid.json`
- `runs/cbl_flip_tta_strict_tiny_gate_valid.json`
- `runs/cbl_flip_tta_scalar_strict_profile_ensemble_valid.json`
- `runs/cbl_flip_tta_strict_tiny_gate_smoke16.json`
- `runs/cbl_flip_tta_scalar_robustness.json`
- `runs/cbl_flip_tta_scalar_selected_predictions.pt`
- `runs/cbl_flip_tta_strict_selected_predictions.pt`

Implementation transport:

- branch `cbl-iterative-depth-20260731`
- commit `91dee77`
- follow-up commit `94c9835`
- ensemble-audit commit `ccac894`

## Related Pages

- [[CBL Refinement Trajectory and Damped Final Step - 2026-07-31]]
- [[CBL Refinement Consistency and Depth Gate - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[Wiki Overview]]
- [[Wiki Log]]
