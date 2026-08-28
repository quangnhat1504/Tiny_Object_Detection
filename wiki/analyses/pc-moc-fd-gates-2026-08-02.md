---
title: PC-MOC-FD Gates - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - wiki/analyses/post-cr-sc-cbl-mechanism-gates-2026-08-02.md
  - runs/rpn_cross_scale_complementarity_full_valid.json
tags: [tiny-object, distillation, fpn, pcgrad, kaggle, validation-only]
---

# PC-MOC-FD Gates - 2026-08-02

## Status

**Gate0, technical integrity, and the seed-2718 performance gate passed.**
Independent checkpoint reload and both original-image folds confirm the gain.
A same-source seed-42 fair-20 baseline/PC-MOC/PC-MR matrix is now running.
No locked-test data was read.

## Method

Projected Micro Object-Centric Feature Distillation (PC-MOC-FD) transfers
cross-scale information only for exact GT micro objects (`sqrt(area) < 8 px`)
where the frozen `960/1200` teacher has top-300 RPN IoU at least `0.50` and
beats the fixed `640/800` student by at least `0.02` IoU.

Each selected GT box is RoI-aligned to a `7x7` FPN feature. Student and teacher
features are normalized across channels and optimized with weighted cosine
distance. The auxiliary recomputes only the student FPN from detached backbone
body features. Its fixed weight is `0.15`, and conflicting gradients are
projected only on student FPN parameters. The teacher and all auxiliary logic
are absent at inference.

This design is motivated by scale-aware small-object distillation in
[ScaleKD](https://openaccess.thecvf.com/content/CVPR2023/html/Zhu_ScaleKD_Distilling_Scale-Aware_Knowledge_in_Small_Object_Detector_CVPR_2023_paper.html)
and task-relevant gradient weighting in
[Gradient-Guided Knowledge Distillation](https://openaccess.thecvf.com/content/WACV2024/html/Lan_Gradient-Guided_Knowledge_Distillation_for_Object_Detectors_WACV_2024_paper.html).
PC-MOC-FD is a project-specific bounded combination, not yet a novelty claim.

## Mechanism Probe

The frozen 20-batch seed-42/batch-4 probe at weight `0.01` selected `83/514`
micro GTs (`16.15%`) with `14/20` valid gradient batches. Raw FPN gradients
conflicted on `11/14` valid batches (`78.57%`), but mean cosine was only
`-0.0181`, and projection retained `99.95%` of the auxiliary norm. The mean
norm ratio was only `0.00382`, so the bounded successor fixed weight `0.15`
before the full Gate0, targeting approximately `0.057`.

Artifact: `runs/moc_fd_fpn_gradient_probe20_seed42.json`.

## Gate0

The preregistered 200-batch audit passed every condition:

| Metric | Result | Gate |
|---|---:|---:|
| valid gradient batches | `158/200 = 79.00%` | `>=60%` |
| selected micro GT | `829/3511 = 23.61%` | `>=12%` |
| raw conflict | `127/158 = 80.38%` | `>=50%` |
| projected cosine | `+0.00468` | `>=0` |
| projected norm ratio | `0.05841` | `[0.03, 0.10]` |
| projected norm retained | `99.91%` | `>=95%` |

Artifact: `runs/pc_moc_fd_fpn_gradient_audit_seed42.json`.

## Technical Gate

Four real batch-size-4 AMP/SGD steps passed. The auxiliary reached the student
FPN but not the backbone body, RPN head, or RoI head. The frozen teacher had
zero gradient parameters and was absent from the student state dict. Inference
before/after attach and after checkpoint reload matched exactly with zero box
and score error. Peak allocated VRAM was `7.254 GiB`.

Artifact: `runs/pc_moc_fd_technical_smoke_seed42.json`.

## Paired Performance Protocol

All seed-2718 arms use two raw/no-EMA epochs, batch size 4, fixed `640/800`
student scale, validation-mAP50 selection of `best.pt`, and identical source
bundle SHA-256
`02c0488ababa2726b48db81165f1ad132d3a6c8483096afb55c4009024b976b1`.
The exact teacher SHA-256 is
`90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a`.

| Arm | Private Kaggle kernel | Status |
|---|---|---|
| baseline | `thyngluthy/tod-icbl-gate2-s2718-r2-20260802` | `COMPLETE` |
| PC-MR-RPN | `hienquang06/tod-pcmr-rpn-gate2-s2718-r2-20260802` | `COMPLETE` |
| PC-MOC-FD | `hngtrngtn/tod-pcmoc-fd-gate2-s2718-20260802` | `COMPLETE` |

All three exact two-T4 smokes completed and passed. The new `hngtrngtn`
credential is the eighth authenticated account in the round-robin pool; no
credential material is recorded here.

Both candidates must independently improve AP and AP75, keep AR100 delta at
least `-0.005`, improve AP on both original-image folds, keep each fold AP75
delta at least `-0.001`, and avoid simultaneous class-aware micro/tiny AP
regression. A serial five-job monitor downloads terminal artifacts, and a
separate local worker runs the prepared audits sequentially.

## Seed-2718 Performance Result

Both downloaded contracts passed, including exact source/teacher hashes, two
metric rows, raw checkpoint source, and validation-mAP50 `best.pt` selection.
Independent baseline to PC-MOC reload changed:

| Metric | Baseline | PC-MOC-FD | Delta |
|---|---:|---:|---:|
| COCO AP | `0.1115` | `0.1178` | `+0.0063` |
| AP50 | `0.3129` | `0.3337` | `+0.0208` |
| AP75 | `0.0501` | `0.0558` | `+0.0057` |
| AR100 | `0.2457` | `0.2651` | `+0.0194` |
| mAP(scale) | `0.5548` | `0.5679` | `+0.0131` |

Even-fold AP/AP50/AP75/AR deltas were
`+0.0088/+0.0239/+0.0054/+0.0226`; odd-fold deltas were
`+0.0022/+0.0149/+0.0043/+0.0153`. Class-aware micro/tiny/small/large changed
`-0.0793/+0.0408/+0.0398/-0.1114`; micro regressed, but tiny improved, so the
frozen micro/tiny guard passed. All six promotion conditions pass.

Artifact: `runs/pc_moc_fd_seed2718_gate_result.json`.

## Fair-20 Promotion

PC-MOC now shares one frozen seed-42, 20-epoch EMA baseline with PC-MR. All
three notebooks use source SHA-256 `e3c1274c...8111`, batch size 4, student
scale `640/800`, validation-mAP50 checkpoint selection, and no locked-test
access. Exact two-T4 smokes and candidate teacher-isolation checks passed.
Long kernels are `ngquangnht/tod-icbl-pcmicro-fair20-s42-20260802`,
`hngngnguynvn/tod-pcmoc-fd-fair20-s42-20260802`, and
`qnhat1504/tod-pcmr-rpn-fair20-s42-20260802`; all are `RUNNING`.

See [[PC Micro Fair-20 Protocol - 2026-08-02]].

## Claim Boundary

Allowed now: PC-MOC has a robust fresh-seed short-schedule validation gain and
is promoted to fair-20 validation.

Not allowed now: a fair-20 gain, a paper checkpoint, or any locked-test
comparison before the frozen 20-epoch artifacts and independent audits pass.

## Related Pages

- [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]]
- [[PC-MHFD Gates - 2026-08-02]]
- [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]]
- [[PC Micro Fair-20 Protocol - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
