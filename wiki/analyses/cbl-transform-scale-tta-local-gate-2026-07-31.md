---
title: CBL Transform-Scale TTA Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources:
  - runs/cbl_scale_tta_scalar_min800_valid.json
  - runs/cbl_scale_tta_strict_min800_valid.json
  - runs/cbl_scale_tta_scalar_min960_valid.json
  - runs/cbl_scale_tta_strict_min960_valid.json
  - runs/cbl_scale_tta_scalar_min960_variants_valid.json
  - runs/cbl_scale960_flip_tta_cache_ensemble_valid.json
  - runs/cbl_scale_tta_scalar_min960_adaptive_unmatched_valid.json
  - runs/cbl_scale_tta_scalar_min960_pair_calibration_valid.json
  - runs/cbl_scale_tta_scalar_min960_pair_calibration_fine_valid.json
  - runs/cbl_scale_tta_scalar_min960_pair_calibration_combined_valid.json
tags: [cbl, localization, tta, scale, fusion, validation]
---

# CBL Transform-Scale TTA Local Gate - 2026-07-31

## Question

Can a larger Faster R-CNN transform scale provide complementary localization
evidence for the fixed CBL EMA checkpoint beyond horizontal flip TTA?

External pre-resizing is not a real scale TTA path here because the detector's
internal transform resizes each tile again. This audit changes
`model.transform.min_size` and `model.transform.max_size` during evaluation,
then pairs detections between the base transform and larger transform.

## Method

`scripts/eval_cbl_scale_tta.py` evaluates the same validation tiles with the
base transform and one larger transform:

1. base view uses the checkpoint's normal transform, `640/800`;
2. scale views tested `800/1000` and `960/1200`;
3. detections are greedily paired by same class at IoU `>=0.50`;
4. paired boxes use score-weighted coordinate averaging and mean score;
5. unmatched base detections are kept, unmatched scale-only detections are
   ignored, then class-aware NMS `0.50` is applied.

The evaluator uses no GT for fusion, does not retrain, and does not change
checkpoint weights. The profile is therefore an offline inference-time gate.

## Full-Validation Gate

The scalar profile is the three-pass refinement leader: passes 1/2 are full,
pass 3 has blend `0.50`, and the score threshold is `0.30`.

| Profile | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Previous flip-TTA scalar pair | 0.1561 | 0.4227 | 0.0785 | 0.2961 |
| Previous strict tiny flip-TTA | 0.1551 | 0.4199 | 0.0795 | 0.2954 |
| Scale 800 scalar pair | 0.1622 | **0.4249** | 0.0868 | 0.3019 |
| Scale 800 strict pair | 0.1622 | 0.4245 | 0.0874 | 0.3020 |
| Scale 960 scalar pair | 0.1629 | 0.4238 | 0.0883 | 0.3024 |
| Scale 960 pair + unmatched scale weight 0.75 | 0.1653 | 0.4303 | 0.0889 | 0.3140 |
| **Size-aware pair, cutoff 12, alpha 0.75/0.40** | **0.1658** | **0.4314** | 0.0892 | **0.3151** |
| **Strict size-aware pair, cutoff 16, alpha 0.85/0.50** | 0.1654 | 0.4286 | **0.0909** | 0.3147 |
| Scale 960 strict pair | 0.1627 | 0.4242 | 0.0872 | 0.3020 |

Transform-scale TTA is a larger gain than horizontal flip TTA. The selected
`960/1200` scalar pair plus low-weight unmatched scale detections improves
the single-view scalar cache from
AP/AP75/AR100=`0.1504/0.0783/0.2943` to the balanced
`0.1658/0.0892/0.3151`. The strict coordinate profile reaches the maximum
AP75=`0.0909`.

The strict center/size damping profile does not beat scalar scale-TTA. It
slightly increases the score-thresholded TP75 count but lowers COCO AP75, so
the scalar profile remains the default high-accuracy setting.

## Robustness

The cache variant audit shows that pair IoU `0.50` is the correct default:

| Scale 960 scalar variant | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Base view | 0.1504 | 0.4073 | 0.0783 | 0.2943 |
| Scale view only | 0.1499 | 0.3873 | 0.0803 | 0.2932 |
| **Pair IoU 0.50** | **0.1629** | **0.4238** | **0.0883** | 0.3024 |
| Pair IoU 0.60 | 0.1615 | 0.4200 | 0.0871 | 0.3009 |
| Pair IoU 0.70 | 0.1587 | 0.4171 | 0.0843 | 0.2992 |
| Pair IoU 0.50 + unmatched scale weight 0.50 | 0.1642 | 0.4276 | 0.0886 | 0.3109 |
| **Pair IoU 0.50 + unmatched scale weight 0.75** | **0.1653** | **0.4303** | **0.0889** | 0.3140 |
| Pair IoU 0.50 + unmatched scale weight 0.90 | 0.1650 | 0.4291 | 0.0887 | 0.3151 |
| Union NMS 0.50 | 0.1553 | 0.4099 | 0.0797 | 0.3093 |
| Union NMS 0.60 | 0.1560 | 0.4086 | 0.0809 | **0.3148** |

Pair IoU `0.50` matched `112,560` of `339,651` base detections, a 33.1%
match rate. The selected unmatched-scale weight `0.75` keeps extra scale-view
detections after multiplying their scores by `0.75`; weight `0.90` gives a
slightly higher AR100 but lower AP/AP75.

The selected unmatched-scale profile repeats on both deterministic validation
halves:

| Fold | Base AP/AP75/AR | Scale pair AP/AP75/AR |
|---|---|---|
| Even tiles | 0.1473 / 0.0788 / 0.2887 | **0.1604 / 0.0870 / 0.3095** |
| Odd tiles | 0.1541 / 0.0776 / 0.2998 | **0.1705 / 0.0920 / 0.3185** |

The larger transform alone increases AP75 but hurts AP50 and background
ranking. Pair fusion is needed to convert the extra high-resolution evidence
into a net AP gain.

## Adaptive Unmatched and Pair Calibration

The unmatched-scale follow-up first tested score floors, predicted-size
filters, size-dependent score weights, and score-dependent weights. No rule
beat constant weight `0.75` on AP or AP75. Filtering original unmatched scale
scores below `0.10` preserves the rounded AP/AP75=`0.1653/0.0889` and TP75
`1,895`, while reducing the audit predictions from `100,416` to `95,864`.
However, exact AP is fractionally lower (`0.1653093` versus `0.1653102`) and
AR100 falls to `0.3138`, so this is only a cleaner optional profile.

A score-step rule using weight `0.50` below score `0.15` and `0.90` above it
reaches AR100=`0.3152`, but lowers AP/AP75 to `0.1650/0.0887`. The gain over
the selected balanced AR is too small to justify another deployment profile.

The positive follow-up changes matched-pair coordinates rather than filtering
detections. Let `alpha` be the scale-view coordinate weight:

`fused_box = (1 - alpha) * base_box + alpha * scale_box`.

For each pair, the gate uses the predicted base-view sqrt-area. Paired scores
remain the mean of the two views, unmatched base detections remain unchanged,
and unmatched scale detections retain score weight `0.75`.

The search used 66 even-indexed original images for tuning and 65 odd-indexed
original images for confirmation. Tiles from the same original image never
cross the fold boundary. This is stronger than the earlier alternating-tile
check.

| Profile | Tune AP/AP75/AR | Confirm AP/AP75/AR | Full AP/AP50/AP75/AR |
|---|---|---|---|
| Previous score-weighted pair | 0.1507 / 0.0802 / 0.2999 | 0.1798 / 0.0988 / 0.3271 | 0.1653 / 0.4303 / 0.0889 / 0.3140 |
| **Balanced: cutoff 12, alpha 0.75/0.40** | **0.1514 / 0.0808 / 0.3011** | **0.1800 / 0.0985 / 0.3280** | **0.1658 / 0.4314 / 0.0892 / 0.3151** |
| **Strict: cutoff 16, alpha 0.85/0.50** | **0.1516 / 0.0826 / 0.3011** | 0.1789 / **0.1001** / 0.3270 | 0.1654 / 0.4286 / **0.0909** / 0.3147 |

The balanced profile improves AP and AR on both image-group folds. Its AP75
gain is not fold-uniform, so it is selected for overall AP, not strict
localization. The strict profile improves AP75 on both folds and is the
AP75-only choice.

Cross-view agreement IoU is informative by itself (IoU75 AUC `0.9014`), but
class score is stronger (AUC `0.9740`). Multiplying scores by agreement powers
does not create a balanced frontier; the strongest shortlisted agreement
variant reaches AP/AP75/AR=`0.1642/0.0900/0.3128`. Keep mean paired scores.

## Scale Plus Flip Ensemble

A cache-only ensemble between the scale-960 pair leader and the earlier
flip-TTA scalar pair produces a new AP/AP50 frontier, but not a new AP75
frontier:

| Profile | AP | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|
| Scale 960 pair + unmatched scale 0.75 | **0.1653** | 0.4303 | **0.0889** | **0.3140** |
| Flip pair | 0.1561 | 0.4227 | 0.0785 | 0.2961 |
| Scale 960 pair-only + flip pair IoU 0.50 | 0.1638 | **0.4320** | 0.0878 | 0.3010 |
| Scale 960 + flip pair IoU 0.60 | 0.1636 | 0.4316 | 0.0876 | 0.3007 |
| Scale 960 + flip pair IoU 0.70 | 0.1635 | 0.4310 | 0.0877 | 0.3007 |

The two cached predictions are highly overlapping: pair IoU `0.50` matches
94.9% of scale-TTA detections. The ensemble improves AP/AP50 by ranking and
coordinate averaging but reduces TP75 in the audit (`1,852 -> 1,823`) and
lowers AP75 on both folds versus scale-960 pair-only. After the unmatched-scale
follow-up, even AP/AP50 no longer beats the selected scale-only cache variant.

## Error Audit

The original scale-960 unmatched-scale profile improves the AP75 audit against
the base scalar cache:

- TP75: `1,709 -> 1,895`
- recall75: `0.2066 -> 0.2290`
- localization FP at IoU 0.50-0.75: `5,612 -> 5,717`
- localization FP at IoU 0.25-0.50: `13,865 -> 14,651`
- background predictions: `70,602 -> 77,930`
- scale TP75: micro `63 -> 101`, tiny `565 -> 661`, small `962 -> 1,005`,
  large `119 -> 128`

The gain is not only a small-object effect; micro, tiny, small, and large TP75
all improve. The cost is more localization/background false positives, so this
remains an offline high-accuracy inference profile rather than a clean
deployment default.

The new pair-coordinate calibration improves the selected unmatched-scale
profile without changing scores:

| Audit profile | TP75 | Loc FP 0.50-0.75 | Loc FP 0.25-0.50 | Background |
|---|---:|---:|---:|---:|
| Previous unmatched weight 0.75 | 1,895 | 5,717 | 14,651 | 77,930 |
| Balanced cutoff 12, alpha 0.75/0.40 | 1,909 | 5,683 | 14,635 | 78,022 |
| Strict cutoff 16, alpha 0.85/0.50 | **1,920** | **5,669** | 14,661 | 78,072 |

Balanced scale TP75 is micro/tiny/small/large=`103/679/1003/124`. Strict scale
TP75 is `106/682/1006/126`. Both shift useful localization evidence toward
micro/tiny objects; strict loses two large TP75 versus the previous profile.

## Cost

The scale-800 full pass processed 1,764 validation tiles in about 201 seconds.
The scale-960 full pass took about 215 seconds with batch size 2 for paired
base+scale generation before COCO evaluation. This is much slower than
horizontal flip TTA, which took about 93 seconds for the same tile count.

## Decision

- Use scalar transform-scale TTA at `960/1200`, base pairing IoU `0.50`,
  unmatched scale score weight `0.75`, and size-aware pair coordinates for
  high-accuracy inference.
- For maximum overall AP, use base predicted-size cutoff `12` px, scale alpha
  `0.75` below the cutoff, and `0.40` above it.
- For maximum AP75, use cutoff `16` px and scale alpha `0.85/0.50`.
- Do not promote the adaptive AR-only score rule. Its AR100=`0.3152` advantage
  is negligible and AP/AP75 are lower.
- Do not promote the current scale+flip ensemble; after unmatched-scale fusion
  it no longer creates a frontier.
- Do not use the strict center/size scale profile; it does not improve the
  selected frontier.
- Do not sweep more transform scales now. `800` and `960` already show the
  main behavior, and pair threshold `0.50` is robust.
- Stop coordinate/score calibration after the bounded broad, fine, and
  combined searches. Further tuning on the same validation set risks
  parameter overfit.
- Do not run Kaggle or reopen the locked test: checkpoint weights are unchanged
  and the CBL family already consumed its frozen locked-test gate.

## Artifacts

- `runs/cbl_scale_tta_scalar_smoke16.json`
- `runs/cbl_scale_tta_scalar_min800_valid.json`
- `runs/cbl_scale_tta_scalar_min800_predictions.pt`
- `runs/cbl_scale_tta_strict_min800_valid.json`
- `runs/cbl_scale_tta_strict_min800_predictions.pt`
- `runs/cbl_scale_tta_scalar_min960_smoke16.json`
- `runs/cbl_scale_tta_scalar_min960_valid.json`
- `runs/cbl_scale_tta_scalar_min960_predictions.pt`
- `runs/cbl_scale_tta_strict_min960_valid.json`
- `runs/cbl_scale_tta_strict_min960_predictions.pt`
- `runs/cbl_scale_tta_scalar_min960_variants_valid.json`
- `runs/cbl_scale960_flip_tta_cache_ensemble_valid.json`
- `runs/cbl_scale_tta_scalar_min960_adaptive_unmatched_valid.json`
- `runs/cbl_scale_tta_scalar_min960_adaptive_unmatched_predictions.pt`
- `runs/cbl_scale_tta_scalar_min960_pair_calibration_valid.json`
- `runs/cbl_scale_tta_scalar_min960_pair_calibration_fine_valid.json`
- `runs/cbl_scale_tta_scalar_min960_pair_calibration_combined_valid.json`
- `runs/cbl_scale_tta_scalar_min960_pair_calibration_fine_predictions.pt`
- `runs/cbl_scale_tta_scalar_min960_sizeaware_balanced_smoke16.json`

Implementation transport:

- branch `cbl-iterative-depth-20260731`
- commit `5cb1034`
- follow-up commit `6745169`
- size-aware calibration commit `af6f2dc`

## Related Pages

- [[CBL Horizontal-Flip TTA Local Gate - 2026-07-31]]
- [[CBL Refinement Trajectory and Damped Final Step - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
- [[Wiki Overview]]
- [[Wiki Log]]
