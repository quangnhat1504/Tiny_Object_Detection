---
title: Confidence-Driven Localization Local Gate - 2026-07-30
type: analysis
created: 2026-07-30
updated: 2026-07-30
sources: [common/model.py, common/config.py, scripts/train_frcnn_metric.py, arxiv:2303.01803]
tags: [ap75, cbl, ugs, distributional-regression, local-gate]
---

# Confidence-Driven Localization Local Gate - 2026-07-30

## Question

Can a classification-based RoI localization head improve strict tiny-box
localization without the instability of CIoU/DIoU or the validation overfit of
the standalone quality-score branch?

## Research Basis

The ICCV 2025 paper
[Uncertainty-Aware Gradient Stabilization for Small Object Detection](https://arxiv.org/abs/2303.01803)
reformulates continuous box regression as classification over an
interval-nonuniform grid. Its earlier C-BBL ablation reports that the R-CNN
stage alone improves Faster R-CNN, while the full method improves VisDrone
Faster R-CNN from 21.5 to 22.9 AP and AP75 from 22.3 to 24.3.

This matches the project diagnosis in
[[Decoupled DFL Regression Plan - 2026-07-06]] but provides a method tested
directly on Faster R-CNN and small-object benchmarks.

## Implementation

The opt-in `--box-loss cbl` path:

- predicts six logits for each class-specific RoI delta coordinate;
- uses a symmetric `[-5, 5]` interval-nonuniform grid with beta 1;
- trains with two-hot cross-entropy plus entropy-matching uncertainty loss;
- restores continuous deltas using the expectation over the full distribution;
- keeps SA-ALW RPN assignment, backbone, tiling, augmentation, and postprocess
  unchanged;
- records all CBL parameters in checkpoint metadata and rebuilds the correct
  predictor during evaluation.

The first gate intentionally excludes the standalone quality-score head and
the uncertainty-guided feature perturbation module. This isolates the
distributional localization change.

## Validation Gate

Run:
`runs/sa_alw_full__cbl__la_loss__seed42__cbl_local_gate1`

Runtime controls: `TOD_USE_EMA=0`, `TOD_NUM_WORKERS=0`,
`TOD_BATCH_SIZE=4`, `box_loss_warmup_epochs=0`.

| Epoch | COCO AP | AP50 | AP75 | AR100 | mAP(scale) | AP micro | Seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1145 | 0.3334 | 0.0454 | 0.2692 | 0.5477 | 0.2898 | 1089.2 |
| 2 | 0.1200 | 0.3523 | 0.0471 | 0.2759 | 0.5512 | 0.3018 | 779.7 |

An independent checkpoint reload reproduced epoch 2 validation at COCO
AP=0.1199, AP75=0.0467, and val loss=1.3095. The small metric difference is
evaluation ordering noise.

## Target-Delta Audit

Before training, exact GT proposals dominated the sampled positive RoIs and
almost all targets were zero. After epoch 2, an audit over 20 batches found:

- 4,276 positive RoIs;
- 72.3% had non-zero regression targets;
- no target coordinate exceeded the configured `[-5, 5]` range;
- non-zero absolute coordinate maxima were 4.48, 4.41, 3.32, and 3.19 for
  `dx`, `dy`, `dw`, and `dh`.

The positive signal is therefore not explained only by learning the zero-delta
GT proposals. The trained RPN supplies meaningful refinement targets and the
configured distribution range does not clip this sample.

## Decision

This is a positive local multi-epoch gate:

- CUDA forward/backward/inference/reload passed;
- train-to-eval-to-train lifecycle stayed stable;
- AP, AP75, AP micro, and AR100 all improved from epoch 1 to epoch 2;
- no locked-test evaluation had been run at the local-gate stage.

Promote exactly one `cbl_full` seed-42 job to the 20-epoch Kaggle budget.
Do not sweep grid range, bin count, uncertainty weight, proposal jitter, or
quality scoring until the full run is downloaded and locally audited.

Promotion requires checking `best.pt`, `best_coco_ap.pt`, and `best_ap75.pt` on
full validation first. Use the locked test once only if the full validation
result exceeds the relevant reference.

## Full-Budget Promotion

The single approved promotion job was launched on 2026-07-30:

- kernel: `quangnhtng/tod-cbl-full-20260730`;
- URL: https://www.kaggle.com/code/quangnhtng/tod-cbl-full-20260730;
- source branch: `cbl-experiments-20260730`;
- pinned source commit: `34ca5c7546d5e700c19069f5119e645f445c8347`;
- run: `cbl_full`, seed 42, 20 epochs, private T4 kernel;
- initial state: `QUEUED`, then `RUNNING` at 2026-07-30 13:41 local time.

The kernel completed successfully and the downloaded artifact audit found all
20 metric rows, the kernel log, and all four checkpoints. The EMA validation
curve peaked at epoch 5:

| Epoch | Stored EMA AP | AP50 | AP75 | AR100 | mAP(scale) |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.1392 | 0.3859 | 0.0639 | 0.2934 | 0.5892 |
| 5 | **0.1440** | 0.3943 | **0.0677** | **0.2959** | **0.6007** |
| 6 | 0.1437 | **0.3963** | 0.0676 | 0.2929 | 0.5990 |
| 8 | 0.1389 | 0.3854 | 0.0649 | 0.2864 | 0.5943 |
| 20 | 0.1075 | 0.3239 | 0.0380 | 0.2386 | 0.5533 |

The run exposed a checkpoint-contract bug: validation used EMA, but legacy
`best*.pt` files saved raw model weights and no EMA state. Therefore the stored
epoch-5 `0.1440/0.0677` AP/AP75 is not reloadable and must not be reported as a
checkpoint result. Independent raw checkpoint reloads gave:

| Raw checkpoint | Epoch | AP | AP50 | AP75 | AR100 | weighted class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| `best_ap75.pt` / `best_coco_ap.pt` | 5 | **0.1277** | **0.3659** | **0.0554** | **0.2768** | **0.5182** |
| `best.pt` | 6 | 0.0940 | 0.2799 | 0.0354 | 0.2399 | 0.4136 |

Raw epoch 5 still passed the validation gate over the two-epoch CBL reference
on AP, AP75, class-aware weighted AP, and class-aware micro AP. The single
authorized locked-test evaluation then produced AP/AP50/AP75/AR100
`0.0987/0.3002/0.0390/0.2486`, with mAP(scale) `0.5723`. This is the best
audited standalone AP75 so far, above Smooth-L1 best `0.0358` and IoU-patches
seed42 `0.0375`. It is not the COCO AP leader because IoU-patches seed2024
reached `0.1002`, and it does not beat the best scale mAP `0.6114`.

Commit `cd84c47` fixes best-checkpoint saving to use the exact evaluated EMA
weights and labels checkpoint/model sources. The 8-epoch recovery kernel
`quangnhtng/tod-cbl-ema8-20260730`, pinned to `40db904`, completed and produced
a reloadable EMA epoch-5 candidate. Independent validation reproduced
AP/AP50/AP75/AR100=`0.1409/0.3891/0.0665/0.2947` and weighted/micro
class-aware AP=`0.5270/0.3697`. All eight rows and four checkpoints passed
artifact audit. This is the current reloadable CBL validation leader; it does
not receive a second locked-test look. See
[[CBL EMA Recovery Audit - 2026-07-30]].

## Paper-Faithful Uncertainty Minimization Ablation

A closer audit of Eq. 12 and Table 6 in the ICCV paper found that official UGS
minimizes prediction entropy directly and uses `lambda=0.5`. The first CBL gate
instead matches prediction entropy to the two-hot target entropy. Both modes
are now explicit and checkpointed:

- `target_match`: preserves the pinned `cbl_full` run;
- `entropy_min`: paper-faithful `L_CE + 0.5 H(p)`.

The paper-faithful mode passed CUDA smoke and completed a two-epoch no-EMA,
no-worker local run after one resume from a post-epoch-1 stall:
`runs/sa_alw_full__cbl__la_loss__seed42__ugs_um05_local_gate1`.

| Mode | Epoch | COCO AP | AP50 | AP75 | AR100 | mAP(scale) | AP micro |
|---|---:|---:|---:|---:|---:|---:|---:|
| `target_match` | 2 | **0.1200** | **0.3523** | **0.0471** | 0.2759 | **0.5512** | **0.3018** |
| `entropy_min`, lambda=0.5 | 2 | 0.1146 | 0.3326 | 0.0440 | **0.2769** | 0.5307 | 0.2255 |

Independent reload of the entropy-minimization checkpoint reproduced
AP=0.1144, AP75=0.0438, AR100=0.2768, and mAP(scale)=0.5308.

Decision: do not promote `entropy_min` to Kaggle. The paper configuration is
not assumed to transfer automatically to this dataset; it loses AP, AP75,
mAP(scale), and especially AP micro despite a small AR100 increase.

## High-Resolution RoI Refinement Ablation

The unexecuted wiki proposal for 14x14 RoIAlign was ported into the current CBL
stack. A direct learned 14-to-7 reducer collapsed classification at epoch 1
(COCO AP/AP75=`0.0027/0.0006`) and was stopped. A safer redesign preserved the
exact standard 7x7 path and added a 14x14 residual behind a zero-initialized
scalar gate.

The gated version passed CUDA smoke, serialization, exact-zero-residual
equivalence, and two local epochs:

| Model | Epoch | COCO AP | AP50 | AP75 | AR100 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| CBL standard | 1 | 0.1145 | 0.3334 | 0.0454 | 0.2692 | 47.4 |
| CBL standard | 2 | **0.1200** | **0.3523** | 0.0471 | **0.2759** | 48.2 |
| CBL + gated RoI14 | 1 | 0.1192 | 0.3240 | **0.0571** | 0.2660 | 20.9 |
| CBL + gated RoI14 | 2 | 0.1145 | 0.3170 | 0.0529 | 0.2686 | 20.8 |

Independent reload reproduced the gated epoch-1 checkpoint at AP=0.1191 and
AP75=0.0571. The learned residual multiplier moved from `-0.01585` at epoch 1
to `-0.00618` at epoch 2 while AP75 also declined, indicating that the model
was closing the expensive branch.

Decision: retain the implementation as an AP75 diagnostic but do not launch a
full Kaggle run before the standard CBL artifact is audited. The best local
checkpoint improves strict localization, but does not improve total AP over
standard CBL and cuts throughput by about 56%.

### Scale-Metric Caveat

During the failed direct reducer run, `mAP(scale)` rose to 0.8179 while COCO AP
collapsed to 0.0027. Code audit found that the custom scale-bin evaluator
matches predictions to ground truths without checking class labels. Historical
`AP_micro/tiny/small/large` and their mean are therefore class-agnostic
diagnostics, not promotion metrics. Use class-aware COCO AP/AP75/AR for current
decisions.

A separately named COCOeval-based custom scale metric was added without
changing historical keys. A synthetic wrong-class perfect-box case scores
legacy AP_micro=1.0 but corrected AP_micro_class_aware=0.0. Re-evaluation gave:

| Checkpoint | micro CA | tiny CA | small CA | large CA | weighted CA |
|---|---:|---:|---:|---:|---:|
| CBL standard epoch 2 | **0.3515** | **0.4417** | **0.5944** | 0.5194 | **0.4938** |
| CBL + gated RoI14 epoch 1 | 0.2490 | 0.4362 | 0.5561 | **0.5501** | 0.4643 |

This confirms that RoI14 trades away the dominant micro bin while improving
strict localization and the large bin; it is not the stronger overall model.

## Distributional RPN Localization Ablation

The paper's stage-specific RPN configuration was implemented independently of
the RoI head on branch `cbl-rpn-20260730`, commit `758e56f`:

- interval-nonuniform RPN delta grid with `alpha=2`, `beta=1`, and grid number
  `n=10` (11 logits per coordinate);
- two-hot cross-entropy with full-distribution expectation decoding;
- SA-ALW hierarchical assignment, anchors, proposal limits, and RoI CBL kept
  fixed;
- RPN uncertainty weight set to zero to isolate classification localization
  before adding paper entropy minimization.

CUDA smoke verified anchor/logit flatten order, finite forward/backward,
inference, and reload. On a real batch, RPN-CBL started with a larger scalar
loss than Smooth-L1 (`2.401` versus `0.332`) but smaller RPN-head gradient norm
(`0.326` versus `0.572`) and smaller backbone gradient norm (`0.00867` versus
`0.01981`), so no arbitrary loss rescaling was introduced. A sampled target
audit covered 195 positive anchors and found zero coordinates outside
`[-2,2]`; the maximum absolute target was `1.339`.

The first epoch was accidentally evaluated through EMA while the standard
local CBL reference had EMA disabled. The EMA metric (`AP/AP75=0.0243/0.0031`)
is therefore a harness confound, not the architecture result. Independent
evaluation of the raw epoch-1 weights gave:

| Model / weights | Epoch | COCO AP | AP50 | AP75 | AR100 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| CBL standard, raw | 1 | **0.1145** | **0.3334** | **0.0454** | **0.2692** | 47.4 |
| CBL + RPN-CBL, raw | 1 | 0.1057 | 0.3011 | 0.0445 | 0.2610 | - |
| CBL standard, raw | 2 | **0.1200** | **0.3523** | **0.0471** | **0.2759** | 48.2 |
| CBL + RPN-CBL, raw | 2 | 0.0971 | 0.2914 | 0.0386 | 0.2600 | 47.3 |

Epoch 2 resumed the raw epoch-1 model and optimizer with EMA disabled.
Performance declined rather than catching the standard RPN. Decision:
RPN-CBL is a negative local performance gate and must not be launched on
Kaggle. Keep the implementation and the EMA lesson, but focus promotion on
the standard RoI-only CBL full-budget artifact.

Artifacts:
`runs/sa_alw_full__cbl__rpn_cbl__la_loss__seed42__cbl_rpn_local_gate1/metrics.csv`
and `runs/cbl_rpn_local_gate1_raw_valid.json`. The epoch-1 CSV row is EMA;
use the raw JSON for the fair epoch-1 comparison.

## Entropy-Aware CBL Score Fusion

Commit `824cef7` added an inference-only ablation that derives class-specific
localization confidence from the normalized entropy of the four CBL coordinate
distributions:

`score = class_probability * (1 - H(p) / log(B))^gamma`.

`gamma=0` preserves the original postprocess. CUDA inference/reload passed with
fusion enabled. Full validation on the standard CBL epoch-2 checkpoint gave:

| Score mode | COCO AP | AP50 | AP75 | AR100 | class-aware weighted AP |
|---|---:|---:|---:|---:|---:|
| Original, `gamma=0` | **0.1200** | **0.3523** | **0.0471** | **0.2759** | **0.4938** |
| Entropy fusion, `gamma=0.1` | 0.1156 | 0.3481 | 0.0418 | 0.2732 | 0.4891 |
| Entropy fusion, `gamma=0.5` | 0.1065 | 0.3293 | 0.0341 | 0.2673 | 0.4692 |

Decision: negative local gate; keep `gamma=0` and do not tune further. With
two-hot target-entropy matching, high entropy can correctly encode a target
between adjacent grid points rather than poor localization. Raw distribution
entropy is therefore not a valid localization-quality score for this model.

Artifacts: `runs/cbl_local_gate1_entropy_score_p01_valid.json` and
`runs/cbl_local_gate1_entropy_score_p05_valid.json`.

## Adversarial RoI Uncertainty Refinement

The paper's uncertainty-guided refinement applies entropy-gradient
perturbations to FPN features, but neither the paper nor its released arXiv
source defines implementation-level `Lur` details or provides code. Commit
`35b28fb` therefore implements an explicitly scoped RoI-level approximation,
not a paper-faithful reproduction:

- compute entropy gradients on positive 1024-D RoI representations;
- add a detached per-RoI L2-normalized perturbation with `rho=0.5`;
- apply a second CBL two-hot CE predictor pass with weight `0.5`;
- leave inference and checkpoint architecture unchanged.

CUDA forward/backward, evaluator `train()+no_grad()`, inference, and reload
passed. A warmed batch-4 probe measured `0.1523 s/step` for standard CBL and
`0.1539 s/step` with RoI-UR, so the extra predictor pass itself is cheap. The
two-epoch no-EMA validation gate gave:

| Model | Epoch | COCO AP | AP50 | AP75 | AR100 | FPS |
|---|---:|---:|---:|---:|---:|---:|
| CBL standard | 1 | **0.1145** | 0.3334 | **0.0454** | **0.2692** | 47.4 |
| CBL + RoI-UR | 1 | 0.1136 | **0.3343** | 0.0432 | 0.2689 | - |
| CBL standard | 2 | **0.1200** | **0.3523** | 0.0471 | 0.2759 | 48.2 |
| CBL + RoI-UR | 2 | 0.1179 | 0.3438 | **0.0485** | **0.2772** | 46.9 |

Independent epoch-2 reload reproduced AP=0.1186, AP75=0.0488, and
AR100=0.2775. However, class-aware weighted scale AP fell from 0.4938 to
0.4755 and class-aware micro AP fell from 0.3515 to 0.2766. The first process
also slowed to about eight seconds per step after epoch-1 validation and
required a clean resume for epoch 2.

Decision: retain RoI-UR as a strict-localization diagnostic only. Its AP75 gain
is small (`+0.0017` on reload), total AP/AP50 and micro performance regress,
and the train-eval-train lifecycle is not clean enough for Kaggle promotion.
Reconsider only after the standard 20-epoch CBL curve is audited.

Artifacts:
`runs/sa_alw_full__cbl__ur0.5_r0.5__la_loss__seed42__cbl_roi_ur_local_gate1/metrics.csv`
and `runs/cbl_roi_ur_local_gate1_best_ap75_valid_reload.json`.

## HBS-RoI Background Smoothing

SET (CVPR 2025) was re-verified from the official paper and code at
`huixinsun/SET@9208fbc4cfe571be4c15dccad8db1665cfdcb9d6`. The released
single-stage FCOS implementation combines FPN background smoothing, loss-gradient
perturbations, and a second dense-head pass. Commit `900199f` implements a
deliberately narrower, non-paper-faithful Faster R-CNN adaptation:

- apply residual HBS convolutions to GT-background regions on FPN levels 0-3,
  using kernels `3/3/5/5`;
- preserve foreground features exactly;
- reuse the sampled proposals/targets for an auxiliary RoI classification and
  CBL localization pass;
- leave RPN and inference on the original features.

CUDA batch-4 forward/backward used about `3.67 GiB`; HBS parameters received
finite gradients, validation under `train()+no_grad()` passed, inference called
the HBS blocks zero times, and checkpoint reload was exact. The two-epoch
no-EMA gate with auxiliary weight `0.5` gave:

| Model | Epoch | COCO AP | AP50 | AP75 | AR100 | weighted class-aware AP |
|---|---:|---:|---:|---:|---:|---:|
| CBL standard | 1 | 0.1145 | **0.3334** | 0.0454 | **0.2692** | - |
| CBL + HBS-RoI 0.5 | 1 | **0.1173** | 0.3239 | **0.0521** | 0.2667 | 0.4495 |
| CBL standard | 2 | **0.1200** | **0.3523** | **0.0471** | **0.2759** | **0.4938** |
| CBL + HBS-RoI 0.5 | 2 | 0.0978 | 0.2941 | 0.0335 | 0.2634 | - |

An independent epoch-1 reload reproduced AP/AP75/AR100
`0.1173/0.0520/0.2666`, but class-aware micro AP was only `0.2508` versus
standard CBL's `0.3515`. Epoch 2 collapsed across all COCO metrics and the
post-validation lifecycle roughly doubled epoch duration.

Decision: weight `0.5` is a negative stability/overall-performance gate despite
the early AP75 signal. Weight `0.1` was then stopped after its first full
validation: AP/AP75/AR100=`0.1165/0.0543/0.2674`, but weighted/micro class-aware
AP fell further to `0.4408/0.2342`. The lower auxiliary weight did not repair
the classification/scale damage it was intended to test. Do not resume or
launch either HBS setting on Kaggle.

Artifacts:
`runs/sa_alw_full__cbl__hbs0.5__la_loss__seed42__cbl_hbs_local_gate1/metrics.csv`
and `runs/cbl_hbs_local_gate1_best_ap75_valid_reload.json`; weight `0.1`:
`runs/sa_alw_full__cbl__hbs0.1__la_loss__seed42__cbl_hbs_w01_local_gate1/metrics.csv`.

## Related Pages

- [[Decoupled DFL Regression Plan - 2026-07-06]]
- [[CIoU/DIoU Decoupled Regression Training Failure — 2026-07-08]]
- [[Test-Set Evaluation — Phase 2 Metrics]]
