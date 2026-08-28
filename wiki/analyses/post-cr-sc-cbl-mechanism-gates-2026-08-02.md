---
title: Post-CR-SC-CBL Mechanism Gates - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - wiki/analyses/coordinate-reliable-sc-cbl-plan-2026-08-01.md
  - cvpr:2024-crosskd
  - neurips:2020-pcgrad
  - neurips:2017-teacher-bounded-detection-kd
  - cvpr:2022-localization-distillation
  - iccv:2023-cfinet
  - cvpr:2020-task-adaptive-distillation
tags: [cbl, distillation, gradient-conflict, cross-head, rpn, micro-object, research-plan]
---

# Post-CR-SC-CBL Mechanism Gates - 2026-08-02

## Status

Three bounded follow-ups were rejected before validation training. A fourth,
**PC-XH-CR-SC-CBL**, passed exact and real-data technical tests but failed its
fresh seed-777 paired performance gate. It is rejected with no cloud run or
parameter sweep. No new method has accessed the locked test.

A fifth bounded follow-up, **RA-CR-SC-CBL**, passed its technical gate but
failed its frozen fresh-seed performance gate. It is rejected without a sweep
or fair-20 promotion. **RA-TB-CBL** is now the active paired Kaggle gate, while
MR-RPN proceeds only through its validation-only Gate0. No new method has
accessed the locked test.

A later spatial-relation FPN successor, **PC-MSDD**, failed its frozen
PCGrad-justification gate and was rejected without a performance run. Its
high-frequency successor **PC-MHFD** passed Gate0, production-integrity, and
exact two-T4 smoke gates; a same-source seed-1618 validation pair is running.
See [[PC-MHFD Gates - 2026-08-02]]. The locked test remains closed.

## Rejected Mechanisms

### Teacher flip consensus

The proposed filter compared the high-resolution teacher's original and
horizontally flipped CBL distributions after exact coordinate realignment.
The 200-batch audit was finite and state-preserving, but retained `98.71%` of
all and tiny-coordinate weight. Teacher disagreement had only `0.5856` AUC for
high coordinate error, and the low-agreement group was not more erroneous than
the high-agreement group by the preregistered margin (`0.1175` versus
`0.1263`). The gate failed; no validation or cloud run is authorized.

Artifact: `runs/cf_cr_sc_cbl_consensus_audit_seed42.json`.

### Ordered-W1 localization distance

Replacing KL with normalized ordered 1-Wasserstein distance respects the
ordered, nonuniform CBL grid. Unit tests verified zero distance for identical
distributions and larger cost for farther mass. The 200-batch audit remained
well aligned with the detector gradient, but the weighted auxiliary/detector
gradient norm ratio was only `0.0107`, below the frozen `0.02` viability floor.
It therefore lacks sufficient optimization signal at the unchanged weight and
is rejected without reweighting or a validation sweep.

Artifact: `runs/ordered_w1_cr_sc_cbl_audit_seed42.json`.

### Direct cross-head distillation

The direct CrossKD-inspired route feeds the student's shared RoI
representation through the frozen teacher localization head. It produces a
stronger localization signal, but its 200-batch box-head audit conflicts with
the detector in `181/200` batches (`90.5%`), with mean cosine `-0.0481` and
mean auxiliary/detector norm ratio `0.1294`. A representation-path smoke also
showed an excessive ratio near `0.59`. Direct summation is rejected because it
would let the auxiliary objective dominate upstream representation learning.

Artifact: `runs/reliable_cross_head_sc_cbl_audit_seed42.json`.

## PC-XH-CR-SC-CBL

The new bounded method preserves the useful cross-head signal while protecting
the detector objective:

1. detach pooled RoI features before the auxiliary recomputation, so the
   distillation loss updates only the student's two-layer RoI box head;
2. map that representation through the frozen teacher CBL distribution head;
3. keep the frozen CR-SC-CBL coordinate reliability weights and KL settings;
4. compute detector and auxiliary gradients on the student RoI box head; and
5. only when their dot product is negative, remove the auxiliary component
   opposing the detector gradient.

This is motivated by
[CrossKD](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html)
and [PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).
It is a project combination and not yet a novelty claim.

### Technical gate

Status: **passed**.

- exact CPU tests preserve aligned gradients and remove exactly the opposing
  auxiliary component;
- `train_one_epoch` integration applies the projected gradient and records
  conflict/cosine/norm telemetry;
- a four-step, batch-size-4 CUDA AMP/SGD smoke completed with distillation loss
  `0.076156`, total loss `4.051583`, and peak allocated VRAM `9.598 GiB`;
- the auxiliary gradient reaches the RoI box head but not the student CBL
  predictor or backbone;
- teacher gradients remain zero; and
- the student serializes, reloads without a teacher, and preserves inference
  boxes, labels, and scores.

### Fresh-seed performance gate

Status: **failed**. The preregistered seed is `777`, unused by prior method
shaping. Baseline and candidate both use two raw/no-EMA epochs, batch size 4,
workers 0, fixed `640/800`, the same iterative-CBL configuration, and
validation-mAP50 `best.pt` selection. After training, the runner independently
reloads both checkpoints on validation and evaluates both original-image
folds.

Pass requires candidate AP and AP75 above the paired baseline, AR100 no more
than `0.005` lower, and no simultaneous class-aware micro/tiny regression. A
failure ends this method without a weight, temperature, or projection sweep.

Both validation-selected checkpoints are raw epoch 1 and independently match
their stored metrics:

| Independent reload | Baseline | PC-XH candidate | Delta |
|---|---:|---:|---:|
| AP | 0.1174 | 0.1162 | -0.0012 |
| AP50 | 0.3292 | 0.3224 | -0.0068 |
| AP75 | 0.0524 | 0.0528 | +0.0004 |
| AR100 | 0.2564 | 0.2585 | +0.0021 |
| mAP(scale) | 0.5700 | 0.5692 | -0.0008 |
| class-aware micro AP | 0.2726 | 0.2719 | -0.0007 |
| class-aware tiny AP | 0.4114 | 0.4207 | +0.0093 |
| class-aware small AP | 0.5767 | 0.5617 | -0.0150 |

The AP75 gain repeats on both folds (`+0.0032` even, `+0.0009` odd), but AP is
not robust (`-0.0047` even, `+0.0027` odd). The primary AP and AP50 conditions
therefore fail despite the small AP75/AR signal. PCGrad was active on only
`2.39%` of epoch-1 and `1.04%` of epoch-2 batches at seed 777, with positive
mean cosines `0.3276/0.3549`; the conflict pattern observed in the seed-42
no-update audit did not transfer to this fresh seed trajectory.

Decision: reject PC-XH-CR-SC-CBL, do not tune its loss/projection settings, do
not launch it on Kaggle, and do not access the locked test. Artifacts:

- `runs/pc_xh_baseline_seed777_valid_reload.json`;
- `runs/pc_xh_candidate_seed777_valid_reload.json`;
- `runs/pc_xh_baseline_seed777_original_image_folds.json`;
- `runs/pc_xh_candidate_seed777_original_image_folds.json`;
- `.runtime/local/pc_xh_cr_sc_cbl_seed777/state.json`.

## RA-CR-SC-CBL

Refinement-Aligned Coordinate-Reliable Scale-Consistent CBL Distillation moves
the existing CR-SC-CBL teacher signal from the first sampled RoI proposal to
the detached proposal produced by the trainable iterative-CBL refinement pass.
The student and frozen high-resolution teacher therefore compare their
class-specific CBL side distributions at the same stage that produced the
project's AP75 gain. The teacher, coordinate-reliability mask, KL temperature,
loss weight, training schedule, and inference graph remain unchanged.

### Technical and gradient gate

Status: **passed**.

- real CUDA batch-size-4, four-step AMP/SGD/reload testing completed with
  distillation loss `0.063062`, total loss `4.039005`, and peak allocated VRAM
  `9.597 GiB`;
- a 200-batch no-update audit produced only `1/200` conflicting gradients
  (`0.5%`), mean cosine `0.1357`, and mean auxiliary/detector norm ratio
  `0.0675`;
- `35,500/48,276` positive coordinates were selected (`73.54%`), with mean
  selected coordinate weight `0.1673`; and
- `99.15%` of positive tiny RoIs retained at least one selected coordinate.

Artifact: `runs/ra_cr_sc_cbl_train_viability_audit_seed42.json`.

### Fresh-seed performance gate

Status: **failed and rejected**. Baseline and RA candidate used seed `9001`, two raw/no-EMA
epochs, batch size 4, fixed `640/800` student scale, identical iterative-CBL
settings, validation-mAP50 `best.pt` selection, and the same embedded source
bundle SHA256 `1ada648e...aa5bca`. The candidate uses the exact frozen fair20
EMA epoch-5 teacher SHA256 `90043edf...769a` at `960/1200`.

Both private smoke notebooks completed on exactly two Tesla T4 GPUs. The RA
smoke had finite distillation/total loss `0.304205/6.753263`, zero teacher
gradients, and no teacher state in the serialized student. The paired kernels
are:

- baseline: `thyngluthy/tod-icbl-gate2-s9001-20260802`;
- candidate: `hienquang06/tod-ra-cr-sc-cbl-gate2-s9001-20260802`.

Both jobs completed with two metrics rows, valid raw checkpoint contracts,
matching source/teacher hashes, and independently reloadable epoch-1
validation-mAP50 checkpoints. Candidate minus baseline on the independent
reload was AP/AP50/AP75/AR100/mAP(scale)=
`+0.0022/+0.0165/-0.0003/+0.0042/+0.0209`. Class-aware micro/tiny/small/large
AP changed `-0.0067/+0.0208/+0.0324/-0.0676`.

The fold audit exposed the failure: even-fold AP/AP75 changed
`-0.0020/-0.0049`, while odd-fold AP/AP75 changed `+0.0062/-0.0001`. The
preregistered gate required independent AP and AP75 gains, AP gains on both
folds, and no fold AP75 drop worse than `-0.001`. RA failed three of those
conditions despite its useful AP50/AR/scale-mAP gains, so it is not promoted.

Artifact: `runs/ra_cr_sc_cbl_seed9001_gate_result.json`.

The locked test remains closed. Before results were visible, the performance
gate was frozen as follows: independent validation reload must improve AP and
AP75; AR100 may not fall by more than `0.005`; AP must improve on both
original-image folds; fold AP75 may not fall by more than `0.001`; and
class-aware micro and tiny AP may not both regress. A downloaded `COMPLETE`
kernel without two metrics rows, valid raw `best.pt`/`last.pt` contracts, exact
hashes, and matching reload metrics is a failed artifact gate rather than
promotion evidence.

## RA-TB-CBL

Refinement-Aligned Teacher-Bounded CBL retains RA's post-refinement proposal
alignment and coordinate-reliability selector, but replaces teacher-logit KL
with the student's exact two-bin CBL target loss on selected coordinates. The
high-resolution teacher therefore acts as a bound: it decides where the
student is still worse, while ground truth remains the optimization target.
This avoids forcing the student toward a teacher distribution that may itself
be biased.

The bounded choice follows the principle in
[Learning Efficient Object Detection Models with Knowledge Distillation](https://proceedings.neurips.cc/paper/2017/hash/e1e32e235eee1f970470a3a6658dfdd5-Abstract.html),
which suppresses extra regression supervision once the student surpasses the
teacher. The use of localization distributions and selective valuable regions
is also consistent with
[Localization Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html).
RA-TB-CBL is a project-specific combination, not yet a novelty claim.

### Technical and gradient gate

Status: **passed**.

- exact target interpolation and coordinate masking tests passed on CUDA;
- real batch-size-4, four-step AMP/SGD/reload testing completed with
  distillation/total loss `0.312761/4.286274` and peak VRAM `9.597 GiB`;
- the 200-batch audit produced `0/200` conflicting gradients, mean cosine
  `0.1121`, and mean auxiliary/detector norm ratio `0.0713`;
- selected-coordinate coverage was `73.54%`, and selected tiny-RoI coverage
  was `99.14%`.

Artifact: `runs/ra_tb_cbl_train_viability_audit_seed42.json`.

### Fresh-seed performance gate

Status: **passed and promoted to fair-20 validation**. A paired seed-`31415` raw/no-EMA two-epoch gate was frozen
with source bundle SHA256 `e54a5284...f12d48`. It uses the same baseline,
teacher, scales, batch size, checkpoint rule, and performance conditions as
the seed-9001 RA gate. Both exact two-T4 smokes passed and the long jobs are:

- baseline: `thyngluthy/tod-icbl-gate2-s31415-20260802`;
- candidate: `hienquang06/tod-ra-tb-cbl-gate2-s31415-20260802`.

Both downloaded artifact contracts passed. Independent baseline to candidate
reload improved AP/AP50/AP75/AR100/mAP(scale) from
`0.1146/0.3049/0.0591/0.2530/0.5067` to
`0.1226/0.3267/0.0646/0.2666/0.5285`, for deltas
`+0.0080/+0.0218/+0.0055/+0.0136/+0.0218`. Class-aware
micro/tiny/small/large AP changed `+0.0421/+0.0441/+0.0024/-0.0393`.

Even-fold AP/AP50/AP75/AR deltas were
`+0.0097/+0.0228/+0.0093/+0.0127`; odd-fold deltas were
`+0.0049/+0.0203/-0.0009/+0.0138`. The odd AP75 result remains inside the
preregistered `-0.001` guard, so all six gates pass. Artifact:
`runs/ra_tb_cbl_seed31415_gate_result.json`.

The next evidence is a same-source seed-42 20-epoch EMA baseline/candidate
pair with validation-mAP50 checkpoint selection. Both exact two-T4 smokes
passed, and the long jobs
`thyngluthy/tod-icbl-fair20-s42-r2-20260802` and
`hienquang06/tod-ra-tb-cbl-fair20-s42-20260802` are now `RUNNING`. Fresh seed
31415 establishes short-schedule robustness but is not yet a full-budget
paper checkpoint. See [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]]. No
locked-test access is authorized.

## Cross-Scale RPN Complementarity and MR-RPN

The RoI-distillation queue remains primary, but a validation-only RPN audit
now provides an evidence-backed pivot if RA and RA-TB both fail. The exact
frozen fair20 EMA epoch-5 checkpoint was evaluated twice over all `1,764`
validation tiles and `8,274` GT instances: once at the student transform
`800/800`, and once at the teacher transform `960/1200`. No weights were
updated and the locked test was not read.

The high-resolution RPN is not a globally better replacement. Overall
top-1500 recall changes from `0.8772 -> 0.8640` at IoU 0.50 and from
`0.3447 -> 0.3343` at IoU 0.75. Its value is concentrated in the `1,927`
micro GT instances below 8 px sqrt-area:

| Proposal cutoff | Base micro R@75 | Teacher micro R@75 | Oracle union R@75 | Teacher rescues | Teacher regressions |
|---|---:|---:|---:|---:|---:|
| top 100 | 0.1162 | 0.1546 | 0.2133 | 187 | 113 |
| top 300 | 0.1515 | 0.1930 | 0.2683 | 225 | 145 |
| top 1500 | 0.1868 | 0.2569 | 0.3513 | 317 | 182 |

At top 1500, the high-resolution teacher wins the per-GT maximum-IoU
comparison on `56.36%` of micro instances and raises mean maximum IoU by
`0.0200`. The positive rescue-minus-regression balance is unique to micro;
tiny, small, large, and overall bands all have negative mean deltas. Artifacts:

- `runs/rpn_scale_teacher_base800_full_valid.json`;
- `runs/rpn_scale_teacher_high960_full_valid.json`;
- `runs/rpn_cross_scale_complementarity_full_valid.json`.

This motivates **Micro-Rescue RPN (MR-RPN)**, a conditional research path.
During training only, the frozen high-resolution teacher identifies micro GTs
whose best proposal is better than the student's. A detached advantage weight
selects one or more student RPN candidates for additional positive-objectness
and box regression supervision, but the regression target remains exact ground
truth rather than the teacher box. This combines the teacher-only-where-better
principle of [Task Adaptive Regularization](https://arxiv.org/abs/2006.13108)
with the dynamic proposal mining motivation of
[CFINet](https://arxiv.org/abs/2308.09534). MR-RPN is a project hypothesis,
not a claim that either cited paper uses this exact mechanism.

Direct MR-RPN failed its early Gate0 probe and is rejected. On 20 exact
batch-size-4 batches, it selected `83/514` micro GTs (`16.15%`) but the joint
objectness/regression auxiliary conflicted on `12/14` valid batches (`85.71%`),
with cosine `-0.0331` and norm ratio `0.6791`. Objectness alone conflicted on
`14/14` batches with cosine `-0.1899`; it must not be used. Regression alone
had positive mean cosine `0.0255`, but still conflicted on `3/14` batches and
was too large at norm ratio `0.9416`. Artifact:
`runs/micro_rescue_rpn_group_probe20_b4.json`.

This evidence freezes one narrower successor, **PC-MR-RPN**. It removes the
objectness auxiliary, uses exact-GT regression only, lowers the fixed loss
weight from `0.05` to `0.005`, and projects only conflicting auxiliary
gradients on the student RPN head. Before implementation, its 200-batch
seed-42/batch-4 no-update gate requires at least `50%` valid-signal batches,
micro selection coverage at least `10%`, raw conflict rate at least `10%` to
justify PCGrad, zero projected conflicts, positive projected mean cosine, and
projected norm ratio in `[0.02, 0.15]`. It also requires zero teacher
gradients/state duplication and later exact default-off inference equivalence.
No Kaggle or locked-test run is authorized at this stage.

### PC-MR-RPN Gate0 and technical gate

Status: **passed**.

The frozen 200-batch seed-42/batch-size-4 audit selected `829/3,511` micro GTs
(`23.61%`) and produced valid regression gradients on `158/200` batches
(`79.0%`). Raw regression gradients conflicted on `50/158` batches (`31.65%`),
which justifies projection. After PCGrad the conflict rate is zero, mean
projected cosine is `0.0197`, and mean projected auxiliary/detector norm ratio
is `0.0849`. Artifact:
`runs/pc_micro_rescue_rpn_gradient_audit_seed42.json`.

The opt-in implementation then passed four real batch-size-4 AMP/SGD steps,
including one raw-conflict step. Peak allocated VRAM was `7.261 GiB`.
Auxiliary gradients reached the RPN head but not the backbone; the frozen
teacher had zero gradient parameters and was absent from the student
`state_dict`. Inference before/after teacher attachment and after checkpoint
reload matched exactly with zero box/score error. Artifact:
`runs/pc_micro_rescue_rpn_technical_smoke_seed42.json`.

### PC-MR-RPN fresh-seed performance gate

Status: **passed and promoted to fair-20 validation**. The paired
baseline and candidate use the identical self-contained source bundle SHA256
`02c0488a...b976b1`, two raw/no-EMA epochs, batch size 4, fixed `640/800`
student scale, validation-mAP50 checkpoint selection, and the exact fair20 EMA
epoch-5 teacher at `960/1200`. The candidate is fixed to regression-only
weight `0.005`, top-300 teacher comparison, `<8 px` micro GTs, teacher IoU
floor `0.50`, advantage margin `0.02`, and RPN-head PCGrad.

Both exact two-T4 smokes passed. Long kernels
`thyngluthy/tod-icbl-gate2-s2718-r2-20260802` and
`hienquang06/tod-pcmr-rpn-gate2-s2718-r2-20260802` completed; the same
baseline also anchors PC-MOC-FD.

Before results, promotion is frozen to the same robust gate as RA-TB:
independent AP and AP75 must improve, AR100 delta must be at least `-0.005`, AP
must improve on both original-image folds, each fold AP75 delta must be at
least `-0.001`, and class-aware micro/tiny AP may not both regress. Both
artifact contracts and independent reloads are mandatory. No locked-test
access is authorized.

Both artifact contracts passed. Independent baseline to PC-MR reload changed
AP/AP50/AP75/AR100/mAP(scale) from
`0.1115/0.3130/0.0501/0.2460/0.5543` to
`0.1212/0.3457/0.0542/0.2600/0.5878`, for deltas
`+0.0097/+0.0327/+0.0041/+0.0140/+0.0335`. Even-fold
AP/AP50/AP75/AR deltas were `+0.0118/+0.0319/+0.0034/+0.0162`; odd-fold
deltas were `+0.0057/+0.0306/+0.0040/+0.0113`. Class-aware
micro/tiny/small/large changed `-0.0660/+0.0560/+0.0603/-0.0989`, so all six
frozen gates pass. Artifact:
`runs/pc_mr_rpn_r2_seed2718_gate_result.json`.

PC-MR now shares the frozen seed-42, 20-epoch EMA fair-20 baseline with
PC-MOC. The source SHA-256 is `e3c1274c...8111`; exact two-T4 smokes passed,
and long kernels `ngquangnht/tod-icbl-pcmicro-fair20-s42-20260802`,
`hngngnguynvn/tod-pcmoc-fd-fair20-s42-20260802`, and
`qnhat1504/tod-pcmr-rpn-fair20-s42-20260802` are `RUNNING`. See
[[PC Micro Fair-20 Protocol - 2026-08-02]]. No fair-20 or locked-test claim
exists yet.

## Related Pages

- [[Coordinate-Reliable SC-CBL Plan - 2026-08-01]]
- [[CR-SC-CBL Multi-Seed Fair-20 Protocol - 2026-08-02]]
- [[Conflict-Aware SC-CBL Plan - 2026-08-01]]
- [[PC-MOC-FD Gates - 2026-08-02]]
- [[PC-MHFD Gates - 2026-08-02]]
- [[PC Micro Fair-20 Protocol - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
