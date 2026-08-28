---
title: PC-MHFD Gates - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - runs/pc_msdd_fpn_gradient_audit_seed42.json
  - runs/pc_mhfd_fpn_gradient_audit_seed42.json
  - runs/pc_mhfd_technical_smoke_seed42.json
tags: [tiny-object, distillation, high-frequency, fpn, pcgrad, kaggle, validation-only]
---

# PC-MHFD Gates - 2026-08-02

## Status

**Gate0 and technical gates passed, but the seed-1618 performance gate
failed.** Aggregate AP/AP75/AR improved, but fold robustness and the
micro/tiny guard failed. PC-MHFD is rejected without a sweep, fair-20,
combination launch, or locked-test access.

## Rejected Predecessor: PC-MSDD

Projected Micro Spatial Dependency Distillation replaced pointwise PC-MOC
matching with a `49x49` within-RoI spatial affinity distribution. This follows
the relational feature-transfer direction in
[G-DetKD](https://openaccess.thecvf.com/content/ICCV2021/html/Yao_G-DetKD_Towards_General_Distillation_Framework_for_Object_Detectors_via_Contrastive_ICCV_2021_paper.html)
and the pixel-relation motivation of
[HS-FPN](https://arxiv.org/abs/2412.10116), but the exact project combination
is only a hypothesis.

Its frozen 200-batch audit had `158/200` valid batches, selected `829/3,511`
micro GTs, reached projected cosine `+0.01282`, norm ratio `0.05038`, and
retained `99.99%` of auxiliary norm. However, raw conflict was only `63/158 =
39.87%`, below the preregistered `>=50%` condition required to justify a
projected-gradient method. PC-MSDD is rejected without changing the gate or
running a performance experiment. This does not prove that spatial-relation
distillation is harmful; it rejects the frozen PC-MSDD formulation.

Artifact: `runs/pc_msdd_fpn_gradient_audit_seed42.json`.

## Method

Projected Micro High-Frequency Distillation (PC-MHFD) preserves the proven
teacher-bounded selector from PC-MOC-FD. It operates only on exact GT micro
objects (`sqrt(area) < 8 px`) where the frozen `960/1200` teacher has top-300
RPN IoU at least `0.50` and beats the fixed `640/800` student by at least
`0.02`.

For every selected `7x7` FPN RoI, student and teacher features are normalized
across channels. A fixed `3x3` average is subtracted to isolate the local
high-frequency residual. The student matches the teacher residual direction
with cosine loss, weighted spatially by the teacher residual energy and then
weighted per object by the detached RPN-IoU advantage. Backbone body features
are detached, the auxiliary updates only the student FPN, and PCGrad removes
only the opposing FPN component. The fixed loss weight is `0.20`; all teacher
and auxiliary state is absent at inference.

The frequency/spatial target is motivated by HS-FPN, which reports that high
frequency and spatial dependency modules enrich tiny-object FPN features. The
teacher-bounded, exact-GT, FPN-only, PCGrad implementation here is a
project-specific combination rather than a novelty claim.

## Probe and Frozen Gate0

The no-update 20-batch seed-42 probe at weight `0.01` selected `83/514` micro
GTs. It had `14/20` valid gradient batches, `78.57%` raw conflict, projected
cosine `+0.00184`, and projected norm ratio `0.00253`. This fixed weight
`0.20` before the 200-batch run, targeting a bounded norm ratio near `0.05`.

The frozen Gate0 then passed every condition:

| Metric | Result | Gate |
|---|---:|---:|
| valid gradient batches | `158/200 = 79.00%` | `>=60%` |
| selected micro GT | `829/3511 = 23.61%` | `>=12%` |
| raw conflict | `129/158 = 81.65%` | `>=50%` |
| projected cosine | `+0.00089` | `>=0` |
| projected norm ratio | `0.04539` | `[0.03, 0.10]` |
| projected norm retained | `99.99%` | `>=95%` |

Artifacts: `runs/pc_mhfd_fpn_gradient_probe20_seed42.json` and
`runs/pc_mhfd_fpn_gradient_audit_seed42.json`.

## Technical Gate

Four real batch-size-4 AMP/SGD steps passed, including one raw-conflict step.
The auxiliary reached the student FPN but not the backbone body, RPN head, or
RoI head. The teacher had zero gradient parameters and was absent from the
student state dict. Inference before/after attach and after checkpoint reload
matched exactly with zero box and score error. Peak allocated VRAM was
`7.254 GiB`.

Artifact: `runs/pc_mhfd_technical_smoke_seed42.json`.

## Paired Performance Protocol

The frozen seed-1618 pair uses two raw/no-EMA epochs, batch size 4, fixed
`640/800` student scale, validation-mAP50 selection of `best.pt`, and identical
self-contained source bundle SHA-256
`2cbf24f43975d4e850fb1051eb3e3433b350f7ba4178b1a7eeb3dce71958eb5a`.
The candidate uses the exact fair20 EMA epoch-5 teacher SHA-256
`90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a`.

Both smokes completed on exactly two Tesla T4 GPUs. The PC-MHFD smoke recorded
finite auxiliary/total loss `0.113452/6.334788`, the exact target and weight,
`100/132` selected micro GTs, PCGrad telemetry, zero teacher gradients, and no
teacher state duplication. Long kernels are:

- baseline: `amongus1504/tod-icbl-gate2-s1618-20260802`;
- candidate: `hngtrngtn/tod-pcmhfd-gate2-s1618-20260802`.

Both completed. Promotion requires independent AP and AP75 gains, AR100
delta at least `-0.005`, AP gains on both original-image folds, every fold
AP75 delta at least `-0.001`, and no simultaneous class-aware micro/tiny AP
regression. Full artifact contracts and independent checkpoint reloads are
mandatory. No locked-test access is authorized.

## Performance Result

Both artifact contracts and independent reloads passed. Baseline to PC-MHFD
changed AP/AP50/AP75/AR100/mAP(scale) from
`0.1018/0.2934/0.0431/0.2327/0.5285` to
`0.1046/0.2971/0.0466/0.2606/0.5256`, for deltas
`+0.0028/+0.0037/+0.0035/+0.0279/-0.0029`.

The aggregate gain is not fold-robust. Even-fold AP/AP50/AP75/AR deltas were
`-0.0012/-0.0166/+0.0066/+0.0249`; odd-fold deltas were
`+0.0048/+0.0241/-0.0046/+0.0294`. Class-aware micro/tiny/small/large changed
`-0.0146/-0.0506/+0.0284/+0.0273`. Therefore both-fold AP, fold AP75 guard,
and micro/tiny-not-both-negative gates fail.

Artifact: `runs/pc_mhfd_seed1618_gate_result.json`.

## Claim Boundary

Allowed now: PC-MHFD is retained as a technically valid negative result that
improves aggregate COCO localization but does not generalize across folds or
the micro/tiny bands.

Not allowed now: a sweep, fair-20 promotion, RA-TB combination run, paper
checkpoint, or any new locked-test comparison.

## Related Pages

- [[PC-MOC-FD Gates - 2026-08-02]]
- [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]]
- [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]]
- [[RA-TB plus PC-MHFD Combination Gates - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
