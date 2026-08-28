---
title: RA-TB plus PC-MHFD Combination Gates - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - runs/ra_tb_pcmhfd_fpn_compatibility_seed42.json
  - runs/ra_tb_pcmhfd_technical_smoke_seed42.json
tags: [cbl, distillation, high-frequency, fpn, pcgrad, combination, validation-only]
---

# RA-TB plus PC-MHFD Combination Gates - 2026-08-02

## Status

**The frozen 200-batch compatibility gate and combined technical contract
passed.** This authorizes a later paired validation implementation only after
PC-MHFD passes its independent seed-1618 performance gate. It is not a cloud
launch, detector-performance result, fair-20 promotion, or locked-test claim.

## Hypothesis

RA-TB improves refined RoI localization by using the high-resolution teacher
only to select coordinates where the student is worse, then optimizing exact
ground-truth CBL targets. PC-MHFD operates at a different stage: it transfers
teacher-energy-weighted local high-frequency residual directions for selected
micro GT RoIs and updates only the student FPN.

The combination keeps both frozen individual weights (`0.25` RA-TB and `0.20`
PC-MHFD). PC-MHFD is the only projected auxiliary. Its PCGrad reference is the
sum of the base detector losses and RA-TB, so the FPN component cannot oppose
the already accepted localization objective. Both paths share the exact same
unregistered teacher object, avoiding duplicate teacher weights or checkpoint
state.

## Frozen Compatibility Gate

Two identically initialized students received the same augmented batch and
per-batch RNG state. One exposed RA-TB; the other exposed PC-MHFD. Detector,
RA-TB, and PC-MHFD gradients were compared on matching student FPN parameter
positions without optimizer updates.

Before the run, pass required exactly 200 batches, joint valid rate `>=60%`,
detector-gradient cosine `>=0.9999` and norm ratio in `[0.999,1.001]`, mean
RA-TB/PC-MHFD cosine `>=-0.10`, detector versus detector-plus-RA cosine
`>=0.95`, nonnegative projected MHFD cosine, projected norm retention `>=95%`,
and final-update cosine/norm versus detector of at least `0.95` and within
`[0.90,1.20]`.

The first full process stopped at batch 81 because two persistent student
graphs caused CUDA allocator fragmentation. Its partial 80-batch artifact is
retained only as failure provenance and is not gate evidence. Explicit tensor
release and periodic cache cleanup fixed the infrastructure issue; the audit
then restarted from batch 1 with the unchanged protocol and completed all 200.

## Compatibility Result

All frozen conditions passed:

| Metric | Result | Gate |
|---|---:|---:|
| exact batches | `200/200` | `200` |
| joint valid batches | `155/200 = 77.50%` | `>=60%` |
| detector-gradient cosine | `0.9999999994` | `>=0.9999` |
| detector-gradient norm ratio | `0.9999999919` | `[0.999,1.001]` |
| RA-TB vs PC-MHFD cosine | `+0.01292` | `>=-0.10` |
| detector vs detector+RA cosine | `0.99557` | `>=0.95` |
| projected MHFD cosine | `+0.00144` | `>=0` |
| projected MHFD norm retained | `99.992%` | `>=95%` |
| detector vs final-update cosine | `0.99469` | `>=0.95` |
| final-update/detector norm ratio | `1.01493` | `[0.90,1.20]` |

RA-TB alone is aligned with the FPN detector gradient (mean cosine `+0.10028`,
only `4%` conflicts). PC-MHFD conflicts with the detector on `80%` of its 155
valid batches, but has a small norm ratio of `0.04462`. Relative to the
detector-plus-RA reference, projection activates on `73.55%` of valid batches
and removes only `0.008%` of mean auxiliary norm. This supports technical
compatibility and the chosen reference; it does not prove an AP gain.

Artifact: `runs/ra_tb_pcmhfd_fpn_compatibility_seed42.json`.

## Combined Technical Contract

The implementation permits FPN micro-feature distillation to share RoI
supervision only when all conditions hold: the exact same teacher object,
PC-MHFD high-frequency target, coordinate-reliable teacher-bounded GT distance,
refined RA stage, and no separate RA PCGrad. Other RoI/FPN combinations remain
rejected.

Four real batch-size-4 AMP/SGD steps passed. One PC-MHFD conflict was projected
against detector-plus-RA. The auxiliary remained FPN-only; the teacher had
zero gradients, was absent from the student state dict, and was not duplicated.
Attach and checkpoint reload preserved inference boxes, labels, and scores
exactly. Peak allocated VRAM was `7.254 GiB`.

Artifact: `runs/ra_tb_pcmhfd_technical_smoke_seed42.json`.

## Next Gate

The independent PC-MHFD seed-1618 candidate failed its frozen gate: even-fold
AP was `-0.0012`, odd-fold AP75 was `-0.0046`, and class-aware micro/tiny AP
both regressed. Therefore the prerequisite is not met and the compatible
RA-TB plus PC-MHFD implementation is **not launched**. Compatibility evidence
is retained as a technical negative checkpoint; no threshold rewrite, sweep,
fair-20, or locked-test access is authorized.

## Related Pages

- [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]]
- [[PC-MHFD Gates - 2026-08-02]]
- [[PC-MOC-FD Gates - 2026-08-02]]
- [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
