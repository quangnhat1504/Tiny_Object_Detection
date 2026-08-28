---
title: CBL Refinement Consistency and Depth Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources:
  - https://openaccess.thecvf.com/content_ECCV_2018/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.html
  - https://openaccess.thecvf.com/content_cvpr_2016/html/Najibi_G-CNN_An_Iterative_CVPR_2016_paper.html
tags: [cbl, localization, iterative-refinement, validation]
---

# CBL Refinement Consistency and Depth Gate - 2026-07-31

## Question

Can movement consistency identify localization quality, and how many shared-head
CBL refinement passes should the trainable EMA checkpoint use?

IoU-Net warns that repeated regression can degrade already well-localized boxes
when classification score does not express localization confidence. G-CNN
provides evidence that iterative regression can work when the regressor learns
a useful path toward the object. The project therefore measured the refinement
trajectory before adding another quality head or heuristic score.

## Paired Full-Validation Diagnosis

`scripts/analyze_refinement_consistency.py` observes detections immediately
before and after one CBL refinement pass. Labels and scores are preserved by
the implementation, so all `137,613` post-NMS detections could be paired with
zero unmatched boxes. The analysis focuses on the `16,462` detections above
the configured refinement score threshold `0.30`.

- mean matched-GT IoU: `0.3192 -> 0.3304`
- boxes crossing upward through IoU75: `319`
- boxes crossing downward through IoU75: `174`
- improved/regressed boxes: `7,685/3,414`
- score-IoU75 AUC: `0.8156`
- self-IoU-IoU75 AUC: `0.6307`
- self-IoU/post-IoU Pearson: `0.0370`

Multiplying class score by self-IoU changes IoU75 AUC only from `0.815575` to
`0.815984` and reduces score-IoU correlation. Refinement consistency is not a
useful localization-confidence replacement and should not be used for
rescoring.

The movement is useful as an optimization diagnosis. Boxes with self-IoU
`0.75-0.90` gain `+0.0202` mean IoU, while boxes at `>=0.97` are effectively
converged. A conditional second-step prototype was therefore tested, but its
ungated control was stronger.

## Depth Gate

All profiles use the exact reloadable EMA epoch-5 checkpoint, score threshold
`0.30`, blend `1.0`, and the full 1,764-tile validation set.

| Profile | AP | AP50 | AP75 | AR100 | class-aware scale AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| 1 step | 0.1486 | 0.4030 | 0.0764 | **0.2949** | 0.5395 | 0.3804 |
| 2 steps, adaptive IoU 0.85 | 0.1491 | 0.4043 | 0.0767 | 0.2941 | 0.5413 | 0.3799 |
| 2 steps, adaptive IoU 0.90 | 0.1494 | 0.4053 | 0.0770 | 0.2932 | 0.5436 | 0.3782 |
| 2 steps, adaptive IoU 0.95 | 0.1498 | 0.4066 | 0.0768 | 0.2938 | 0.5438 | 0.3773 |
| 2 steps, ungated | 0.1500 | **0.4075** | 0.0770 | 0.2942 | 0.5440 | 0.3775 |
| **3 steps, ungated** | **0.1501** | 0.4074 | **0.0774** | 0.2934 | 0.5461 | 0.3818 |
| 4 steps, ungated | 0.1500 | 0.4076 | 0.0766 | 0.2921 | **0.5474** | **0.3824** |

The fourth pass is past the COCO localization optimum: AP75 and AR decline.
The adaptive gate also loses to ordinary two-step refinement, so its temporary
core implementation was removed.

## Scale-Aware Extra Passes

The ungated three-step audit gains `40` small-object TP75 but loses `17`
tiny-object TP75 relative to one step. A scale-aware gate therefore keeps pass
one unchanged and applies passes two/three only when predicted
`sqrt(box area) / sqrt(image area)` reaches a minimum ratio.

| Extra-pass cutoff | AP | AP50 | AP75 | AR100 | class-aware scale AP | micro class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| None, three steps | 0.1501 | 0.4074 | **0.0774** | 0.2934 | **0.5461** | 0.3818 |
| 16 px equivalent (`0.03125`) | 0.1501 | **0.4082** | 0.0770 | 0.2945 | 0.5441 | 0.3806 |
| **12 px equivalent (`0.0234375`)** | **0.1504** | 0.4081 | 0.0772 | **0.2946** | 0.5449 | **0.3831** |

The 12 px cutoff is the new overall-AP profile. Its AP75 audit raises TP75
from `1,728` to `1,740` relative to ungated three-step refinement, restores
tiny TP75 from `565` to `580`, preserves small TP75 (`978 -> 977`), and lowers
background predictions from `70,722` to `70,560`. Audit recall75 rises from
`0.2088` to `0.2103`; audit AP75 is `0.0787` versus the ungated `0.0788`.

The AP75 error audit confirms that the three-step result is structured rather
than a summary-metric fluctuation:

- TP75: `1,702 -> 1,728`
- recall75: `0.2057 -> 0.2088`
- localization FP at IoU `0.50-0.75`: `5,893 -> 5,514`
- localization FP at IoU `0.25-0.50`: `14,407 -> 13,768`
- scale TP75 change: micro `+2`, tiny `-17`, small `+40`, large `+1`

On one real tiny-object tile after warmup, one/two/three steps measure
`45.5/42.0/38.9 FPS`. Three steps cost about 14.6% throughput versus one.

## Final-Step Damping Follow-Up

A full trajectory audit found that damping only the third update to `0.50`
improves the independent standard evaluator to
AP/AP50/AP75/AR100=`0.1505/0.4077/0.0781/0.2943`. Combining the damping with
the 12 px gate reaches `0.1505/0.4072/0.0779/0.2953`. These supersede the
earlier AP/AP75 and AR leaders respectively; see
[[CBL Refinement Trajectory and Damped Final Step - 2026-07-31]].

## Decision

- Use three ungated passes with final-step blend `0.50` for maximum standard
  COCO AP/AP75.
- Add the extra-pass minimum size ratio `0.0234375` with final-step blend
  `0.50` for maximum AR and the best AP/AR tradeoff.
- Keep the full-update three-pass profile when class-aware scale AP is primary.
- Use two ungated passes when throughput is weighted more heavily.
- Stop at three passes for strict localization; do not continue a depth sweep.
- Do not add self-IoU score calibration or keep the adaptive gate.
- No Kaggle run or locked-test look is needed: the weights are unchanged and
  the CBL family has already consumed its test gate.
- A follow-up three-step unrolled training ablation was negative. Keep
  one-pass training and apply depth only at inference; see
  [[Unrolled Iterative CBL Training Local Gate - 2026-07-31]].

## Artifacts

- `runs/refinement_consistency_ema8_ep5_valid.json`
- `runs/refinement_consistency_ema8_ep5_valid.csv`
- `runs/cbl_iterative_train_ema8_step2_ungated_valid.json`
- `runs/cbl_iterative_train_ema8_step3_ungated_valid.json`
- `runs/cbl_iterative_train_ema8_step4_ungated_valid.json`
- `runs/cbl_iterative_train_ema8_adaptive_step2_iou085_valid.json`
- `runs/cbl_iterative_train_ema8_adaptive_step2_iou090_valid.json`
- `runs/cbl_iterative_train_ema8_adaptive_step2_iou095_valid.json`
- `runs/cbl_iterative_train_ema8_step3_extra_min12_valid.json`
- `runs/cbl_iterative_train_ema8_step3_extra_min16_valid.json`
- `runs/ap75_analysis_cbl_iterative_train_ema8_ep5_step3_valid/summary.json`
- `runs/ap75_analysis_cbl_iterative_train_ema8_ep5_step3_extra_min12_valid/summary.json`

Implementation transport:

- branch `cbl-iterative-depth-20260731`
- diagnostic commit `2bab8ce`
- scale-aware depth commit `fb0fad4`

## Related Pages

- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[Iterative CBL Refinement Gate - 2026-07-31]]
- [[Unrolled Iterative CBL Training Local Gate - 2026-07-31]]
- [[CBL Refinement Trajectory and Damped Final Step - 2026-07-31]]
- [[Wiki Overview]]
- [[Wiki Log]]
