---
title: Cross-Scale CBL Localization Distillation Plan - 2026-08-01
type: analysis
created: 2026-08-01
updated: 2026-08-01
sources:
  - paper/checkpoints/iterative_cbl_fair20_2026-08-01.md
  - wiki/analyses/cbl-transform-scale-tta-local-gate-2026-07-31.md
  - wiki/analyses/cbl-stochastic-multiscale-training-local-gate-2026-07-31.md
  - common/model.py
  - scripts/train_frcnn_metric.py
tags: [cbl, distillation, cross-scale, localization, tiny-object, research-plan]
---

# Cross-Scale CBL Localization Distillation Plan - 2026-08-01

## Status

Working method name: **Scale-Consistent CBL Distillation (SC-CBL)**.

This is a validation-only research candidate, not a paper headline or novelty
claim. The distributional loss, CUDA path, and first two-epoch gate are
complete. The gate is positive on all primary COCO metrics, but micro/tiny
class-aware diagnostics regress. No Kaggle run, external benchmark, or
locked-test evaluation has been completed.

The frozen paper checkpoint remains [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]].
Its 65-image locked-test budget is consumed `1/1` and must not be reopened for
SC-CBL.

## Motivation From Project Evidence

The strongest unused training signal is the repeatable `960/1200` transform
view in [[CBL Transform-Scale TTA Local Gate - 2026-07-31]]:

- base single-view validation AP/AP75/AR100 is `0.1504/0.0783/0.2943`;
- scale-pair fusion reaches `0.1658/0.0892/0.3151`;
- the gain repeats on both original-image validation folds; and
- the high-resolution view alone raises AP75 but lowers AP50, showing
  complementary localization evidence rather than a uniformly better detector.

Naive stochastic multi-scale training fails at
`0.1141/0.0436/0.2590` versus the fixed-scale two-epoch baseline
`0.1269/0.0572/0.2758`. Correct SNIP-like ignored-object supervision also
fails. SC-CBL therefore does not mix scale tasks in the student. It keeps the
student transform fixed at `640/800` and transfers only useful localization
distributions from a frozen `960/1200` teacher.

## Related Work Boundary

- [Localization Distillation, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)
  motivates KL transfer of localization probability distributions rather than
  only feature imitation.
- [ScaleKD, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html)
  shows that small-object detection benefits from scale-decoupled teacher
  knowledge and a cross-scale assistant.
- [Multi-Scale Aligned Distillation, CVPR 2021](https://arxiv.org/abs/2109.06875)
  aligns feature-pyramid levels between high-resolution teachers and
  low-resolution students.
- [Spatial Self-Distillation, ICCV 2023](https://arxiv.org/abs/2307.12101)
  uses self-distillation to refine inaccurate boxes, including tiny boxes.
- [UGS/C-BBL](https://arxiv.org/abs/2303.01803) supplies the
  interval-nonuniform coordinate distributions already used by this project.

SC-CBL does not claim feature distillation, CBL, or localization distillation
as new. The project-specific hypothesis is that the existing CBL coordinate
distributions provide a natural aligned interface for transferring only the
high-resolution teacher updates that improve ground-truth IoU on the same RoI.

## Proposed Method

For each sampled positive RoI, use the same original-image proposal geometry
in two views:

1. student: trainable SA-ALW-ICBL at fixed `640/800`;
2. teacher: frozen fair20 EMA epoch-5 checkpoint at `960/1200`;
3. project the student proposal to teacher coordinates using exact image-shape
   ratios;
4. select the foreground-class CBL logits `[4, 6]` in both views;
5. decode both predicted boxes and compare aligned IoU to the training GT;
6. distill only if `IoU_teacher >= IoU_student + 0.02`; and
7. weight micro/tiny RoIs by `clamp(16 / sqrt(area_px), 1, 2)`.

The auxiliary objective is:

`L_SC = lambda * w_size * T^2 * mean_coord KL(p_teacher^T || p_student^T)`

with the first bounded configuration `lambda=0.25`, `T=2`, advantage margin
`0.02`, and teacher transform `960/1200`. The teacher is detached and frozen.
Classification, RPN, SA-ALW assignment, base CBL loss, iterative-refinement
loss, and student inference remain unchanged.

Unlike entropy-based selection, the gate uses training GT to measure actual
teacher advantage. This choice is deliberate because the previous project
entropy/score-quality routes failed to predict localization quality reliably.
GT is used only while training and creates no inference dependency.

## Implemented Surface

- `common/model.py`: temperature-scaled CBL KL loss, paired teacher-advantage
  mask, scale-aligned positive-RoI teacher path, and opt-in teacher attachment.
- `scripts/train_frcnn_metric.py`: checkpointed SC-CBL CLI configuration.
- `common/train_utils.py`: distillation-loss accounting.
- `scripts/test_cbl_cross_scale_distillation.py`: unit and gradient contract.
- `scripts/test_cbl_cross_scale_teacher.py`: real-data two-model CUDA smoke.

Technical evidence:

- CUDA unit loss `0.541817`, with student-only gradients and correct masks;
- fair20 real-data scale-alignment/backward loss `0.044578` at margin `0.02`;
- batch-size-4 four-step AMP/SGD smoke ends at distillation/total loss
  `0.052765/4.026421`, with peak allocated memory `6.170 GiB`; and
- no teacher parameters registered in the student optimizer or checkpoint.

## Validation Protocol

### Gate 1: stability

Status: **passed**. Several optimizer steps with batch sizes 1 and 4 produced
no OOM, NaN, skipped batch, teacher gradient, state-dict duplication, or
inference change.
Peak allocated VRAM was `5.053 GiB` at batch 1 and `6.170 GiB` at batch 4.

### Gate 2: two-epoch local comparison

Status: **passed, with a scale-band caveat**. Use seed 42, fixed `640/800`
student training and validation, raw weights,
the existing copy-paste and tiny-tile oversampling, and exactly two epochs.
Compare with the already reloaded fixed-scale leader:

| Method | AP | AP50 | AP75 | AR100 | mAP(scale) |
|---|---:|---:|---:|---:|---:|
| Fixed-scale SA-ALW-ICBL | 0.1269 | 0.3612 | 0.0572 | 0.2758 | 0.5903 |
| SC-CBL, epoch-2 reload | **0.1287** | **0.3628** | **0.0586** | **0.2765** | **0.5910** |

Absolute deltas are `+0.0018/+0.0016/+0.0014/+0.0007/+0.0007`, or about
`+1.42%/+0.44%/+2.45%/+0.25%/+0.12%` relative. Epoch 1 is also positive
against the matching epoch-1 baseline:
`0.1220/0.3478/0.0556/0.2735` versus
`0.1161/0.3328/0.0498/0.2613`.

The exact raw epoch-2 `best.pt` reload matches the stored row within evaluator
rounding. It also improves COCO AP_small/medium/large from
`0.1046/0.2348/0.3411` to `0.1057/0.2393/0.3555`, total TP from `5,828` to
`5,951`, precision from `0.0628` to `0.0651`, and recall from `0.7044` to
`0.7192`.

The caveat is the legacy scale diagnostic. Micro/tiny AP changes
`0.3400/0.6197 -> 0.3260/0.5938`, while small/large improves
`0.6267/0.6974 -> 0.6554/0.7837`. The full-gradient SC-CBL path is therefore a
small overall positive, not yet a clean tiny-object improvement.

Original-image fold robustness is mixed:

| Fold | Baseline AP/AP50/AP75/AR | SC-CBL AP/AP50/AP75/AR | Delta AP/AP75/AR |
|---|---|---|---|
| Even, 66 images | .1128/.3279/.0482/.2596 | .1163/.3390/.0499/.2609 | +.0035/+.0017/+.0013 |
| Odd, 65 images | .1432/.3964/.0700/.2913 | .1413/.3875/.0686/.2924 | -.0019/-.0014/+.0011 |

AR100 improves on both folds, but AP/AP50/AP75 gains do not repeat on the odd
fold. The full-validation gain is real for this evaluator/checkpoint pair but
is not fold-robust enough for a full-budget cloud promotion.

Pass requires a Pareto-positive AP/AP75 result without an AR100 collapse.
The promotion target is AP `>0.1269`, AP75 `>0.0572`, and AR100 no lower than
`0.2708` (`-0.005`). Also report micro/tiny class-aware AP and both
original-image validation halves.

Only the preregistered default configuration was eligible for this gate. Do
not sweep temperature, weight, margin, or teacher scale on the same validation
set after seeing the result.

One structural follow-up is preregistered from the observed failure mode:
**head-only SC-CBL** detaches the sampled student RoI feature before the
distillation predictor, so KL updates only `bbox_dist` and cannot shift the
shared box head/backbone. All numeric settings and the two-epoch schedule stay
fixed. It passes the isolated-gradient and four-step CUDA gate. This is the
only authorized follow-up before the Kaggle decision; a failure ends the
isolation branch without further local variants.

Result: **head-only rejected**. Epoch-2 reload gives
AP/AP50/AP75/AR100=`0.1124/0.3418/0.0443/0.2626`, and legacy
micro/tiny/small/large=`0.3073/0.5881/0.5802/0.7291`. Updating only the final
distribution predictor is insufficient; the small positive full-gradient
result depends on adaptation of the shared RoI representation. Do not test a
third local SC-CBL variant from this validation result.

### Gate 3: cloud and paper evidence

Cloud decision: **do not promote the current SC-CBL configuration**. The
full-gradient result wins the two-epoch aggregate gate, but loses AP/AP75 on
one original-image fold and regresses micro/tiny diagnostics; head-only fails
outright. Before any 20-epoch Kaggle run, require a new preregistered method or
independent seed evidence that addresses the scale-band/fold inconsistency.
Future paper evidence must use seeds 42/123/2024, an external public benchmark,
and disclose the extra teacher training FLOPs and memory.
The current locked test remains closed. A new external test protocol is needed
before any paper test claim.

## Claim Boundary

Allowed now:

- a technically valid cross-scale CBL distillation prototype;
- high-resolution teacher and base-scale student RoIs are geometrically
  aligned;
- the auxiliary path adds training cost but no inference path; and
- one two-epoch seed-42 validation gate improves all primary COCO metrics over
  the matching fixed-scale baseline.

Not allowed now:

- SC-CBL improves locked-test performance;
- the AP/AP75 gain repeats on both original-image validation folds;
- SC-CBL is novel relative to all localization-distillation work;
- the method generalizes beyond the maritime dataset; or
- the method is ready for Kaggle or paper headline results.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__csldw0.25t2m0.02__la_loss__seed42__cbl_sc_distill_local_gate/metrics.csv`
- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__csldw0.25t2m0.02__la_loss__seed42__cbl_sc_distill_local_gate/best.pt`
- `runs/cbl_sc_distill_best_map50_valid_reload.json`
- `runs/cbl_iterative_train_local_gate_original_image_folds.json`
- `runs/cbl_sc_distill_original_image_folds.json`
- `runs/cbl_sc_distill_head_only_best_map50_valid_reload.json`

## Related Pages

- [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]]
- [[Conflict-Aware SC-CBL Plan - 2026-08-01]]
- [[CBL Transform-Scale TTA Local Gate - 2026-07-31]]
- [[CBL Stochastic Multi-Scale Training Local Gate - 2026-07-31]]
- [[CBL SNIP-Like Scale-Normalized Training Local Gate - 2026-07-31]]
- [[Wiki Overview]]
- [[Wiki Log]]
