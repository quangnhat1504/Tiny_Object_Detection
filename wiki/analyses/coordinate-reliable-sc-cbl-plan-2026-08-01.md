---
title: Coordinate-Reliable SC-CBL Plan - 2026-08-01
type: analysis
created: 2026-08-01
updated: 2026-08-02
sources:
  - wiki/analyses/cross-scale-cbl-localization-distillation-plan-2026-08-01.md
  - wiki/analyses/conflict-aware-sc-cbl-plan-2026-08-01.md
  - cvpr:2022-localization-distillation
  - arxiv:2406.06999
tags: [cbl, distillation, localization, uncertainty, research-plan]
---

# Coordinate-Reliable SC-CBL Plan - 2026-08-01

## Status

Gates A, B, and C passed. The seed-123 fair-20 candidate remains `RUNNING`, and
a frozen three-seed paired fair-20 matrix now adds seed-42 and seed-2024
evidence plus the missing seed-123 baseline. Working name:
**Coordinate-Reliable Scale-Consistent CBL Distillation (CR-SC-CBL)**. This is
a promoted validation candidate, not yet a paper checkpoint or locked-test
performance claim.

## Evidence and Pivot

The original SC-CBL aggregate result is positive but not fold robust. Its
head-only variant fails, so shared RoI representation adaptation is necessary.
The CA-SC-CBL audit then rejects gradient conflict as the explanation: only
`2.0%` of 200 batches have negative detector/distillation cosine, while mean
cosine is `0.1497`.

The remaining weakness is the supervision unit. SC-CBL gates an entire RoI
using decoded-box IoU, then transfers all four coordinate distributions. A
teacher can improve overall IoU while still being worse or uncertain on one or
more individual sides, especially for tiny boxes where a subpixel side error
changes IoU sharply.

[Localization Distillation](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)
supports selective distributional localization transfer. [Teaching with
Uncertainty](https://arxiv.org/abs/2406.06999) argues that imperfect teacher
knowledge should be uncertainty-aware. CR-SC-CBL uses these as related-work
motivation; coordinate-level GT advantage from the existing CBL distributions
is a project hypothesis, not a novelty claim.

## Proposed Objective

Keep the full SC-CBL student/teacher scales, aligned RoIs, shared-head gradient
path, temperature `2`, loss weight `0.25`, and tiny-object weight unchanged.
For RoI `i` and coordinate `j`, decode student and teacher distribution
expectations in box-delta space and compute:

`a_ij = clamp((|dS_ij-t_ij| - |dT_ij-t_ij|) / (|dS_ij-t_ij| + |dT_ij-t_ij| + eps), 0, 1)`

`c_ij = clamp(1 - H(pT_ij) / log(B), 0, 1)`

`w_ij = stopgrad(tiny_weight_i * a_ij * c_ij)`

Normalize coordinate KL by the sum of nonzero `w_ij`. This has no new tuned
threshold: a coordinate contributes only when the teacher expectation is
closer to its exact training target, and its influence falls continuously with
teacher entropy or vanishing advantage. Detaching the weights prevents the
student from changing the gate instead of reducing KL.

## Preregistered Gates

### Gate A: train-only viability

Run 200 deterministic seed-42 batches at the same initialization and data
protocol as the CA-SC-CBL audit, with no optimizer update. Require:

- 200 finite detector and auxiliary gradient pairs;
- selected-coordinate coverage between `5%` and `95%`;
- positive finite coordinate-weight sum and auxiliary gradient norm; and
- no teacher gradients, state-dict duplication, or inference-path change.

This gate may describe the selection mechanism but cannot compare validation
performance.

Status: **passed**. Across 200/200 finite batch pairs, CR-SC-CBL selected
`30,909/48,284` coordinates (`64.01%`) from `12,071` positive RoIs. The mean
nonzero coordinate reliability weight was `0.2652`. The weighted auxiliary
gradient norm averaged `0.0778` of the detector gradient and its mean cosine
was `0.1630`. No optimizer update or validation access occurred.

Artifact: `runs/cr_sc_cbl_train_viability_audit_seed42.json`.

### Gate B: technical optimization

Require exact coordinate-mask and zero-weight tests, teacher-detach tests, AMP
batch-size-4 four-step optimization, checkpoint reload, and default-off
behavior identical to SC-CBL.

Status: **passed**. Unit tests verified exact coordinate masking, zero-weight
behavior, detached teacher gradients, and two reliable synthetic coordinates.
The real-data CUDA integration completed four AMP optimizer steps at batch size
4 with finite scale loss `0.076201`, total loss `4.048642`, and peak allocated
VRAM `9.597 GiB`. The serialized student reloaded without a teacher and matched
the attached model's evaluation boxes, scores, and labels.

### Gate C: fresh-seed paired performance

If Gates A and B pass, use seed `123` for exactly two fresh epochs:

1. fixed-scale SA-ALW-ICBL baseline; and
2. CR-SC-CBL with the frozen SC-CBL numeric configuration.

Pass only if AP and AP75 improve, AR100 falls by no more than `0.005`, and
micro/tiny diagnostics do not both regress. Report both original-image folds.
No seed-42 validation shaping, parameter sweep, Kaggle run, or locked-test
access is authorized until this gate passes.

Status: **passed**. Both valid runs were restarted from scratch with seed
`123`, raw weights, no EMA, batch size 4, workers 0, fixed `640/800` student
scale, and exactly two epochs. The first baseline attempt suffered a CUDA
paging slowdown after epoch 1 and is invalid; it is retained only as an
infrastructure-failure record. The valid comparison uses the restart tagged
`baseline_ec1` against `candidate_ec1`.

| Independent reload | Baseline | CR-SC-CBL | Absolute delta |
|---|---:|---:|---:|
| AP | 0.1133 | 0.1203 | +0.0070 |
| AP50 | 0.3149 | 0.3311 | +0.0162 |
| AP75 | 0.0540 | 0.0570 | +0.0030 |
| AR100 | 0.2599 | 0.2644 | +0.0045 |
| mAP(scale) | 0.532875 | 0.544111 | +0.011236 |
| Class-aware micro AP | 0.2717 | 0.2943 | +0.0226 |
| Class-aware tiny AP | 0.4068 | 0.4193 | +0.0125 |
| Class-aware small AP | 0.5410 | 0.5531 | +0.0121 |

Legacy micro AP regresses `0.2598 -> 0.2504`, but legacy tiny and small AP
increase `0.5300 -> 0.5463` and `0.6079 -> 0.6214`; therefore the preregistered
micro/tiny guard passes. Precision, recall, and matched TP also improve from
`0.0516/0.6894/5704` to `0.0529/0.6917/5723`.

The original-image folds confirm that AP, AP50, and AR improve independently:

| Fold | AP delta | AP50 delta | AP75 delta | AR100 delta |
|---|---:|---:|---:|---:|
| Even | +0.0062 | +0.0124 | +0.0068 | +0.0059 |
| Odd | +0.0075 | +0.0194 | -0.0006 | +0.0032 |

AP75 is strongly positive on the even fold and essentially flat/slightly
negative on the odd fold, so the cloud run must establish whether the full
schedule makes strict localization robust. Artifacts:

- `runs/cr_sc_cbl_seed123_baseline_ec1_reload_and_folds.json`
- `runs/cr_sc_cbl_seed123_candidate_ec1_reload_and_folds.json`

## Kaggle Fair-20 Promotion

The private smoke kernel
`quangnhtng/tod-cr-sc-cbl-smoke-20260801` completed on two Tesla T4 GPUs. It
mounted the frozen fair20 EMA epoch-5 teacher, produced finite total and
distillation losses `6.6278/0.2861`, found zero teacher gradients, and confirmed
that no teacher keys enter the student state dict.

The full private kernel
`quangnhtng/tod-cr-sc-cbl-fair20-20260801` version 1 is `RUNNING`. Its protocol
is seed `123`, 20 fresh epochs, EMA, the same SGD/cosine schedule and iterative
CBL settings as fair20, the frozen `960/1200` teacher, and validation-mAP50-only
selection of `best.pt`. The notebook is self-contained and records source
bundle SHA-256
`6df703015677a50860a5a9c3c4ae3fad5f5f4c281b8a03668781bf7420fb5c5d`.

No locked-test source is mounted. A `RUNNING` kernel is pending evidence: the
method is not promoted to a paper checkpoint until outputs are downloaded,
`metrics.json` and failure artifacts are inspected, and `best.pt` is
independently reloaded on validation. The already consumed locked-test budget
remains closed.

On 2026-08-02, seven configured Kaggle accounts were authenticated and four
additional private fair-20 kernels passed two-T4 smokes before launch. The
complete seed `42/123/2024` pairing, exact teacher hash, kernel IDs, artifact
gate, and serial monitor are frozen in
[[CR-SC-CBL Multi-Seed Fair-20 Protocol - 2026-08-02]]. This expansion changes
only replication evidence, not the method or its hyperparameters.

## Related Pages

- [[Cross-Scale CBL Localization Distillation Plan - 2026-08-01]]
- [[Conflict-Aware SC-CBL Plan - 2026-08-01]]
- [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]]
- [[CR-SC-CBL Multi-Seed Fair-20 Protocol - 2026-08-02]]
- [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
