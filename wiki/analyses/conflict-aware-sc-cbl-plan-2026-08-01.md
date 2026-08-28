---
title: Conflict-Aware SC-CBL Plan - 2026-08-01
type: analysis
created: 2026-08-01
updated: 2026-08-01
sources:
  - wiki/analyses/cross-scale-cbl-localization-distillation-plan-2026-08-01.md
  - neurips:2020-pcgrad
  - neurips:2021-cagrad
tags: [cbl, distillation, gradient-conflict, pcgrad, research-plan]
---

# Conflict-Aware SC-CBL Plan - 2026-08-01

## Status

Rejected at Gate 0. The preregistered 200-batch audit found only `4/200`
negative-cosine batches (`2.0%`), below the `10%` continuation threshold. No
gradient projection or performance run is authorized.
Working name: **Conflict-Aware Scale-Consistent CBL Distillation
(CA-SC-CBL)**.

This direction follows the completed
[[Cross-Scale CBL Localization Distillation Plan - 2026-08-01]] and does not
reopen its hyperparameter sweep. The fair20 locked test remains closed.

## Evidence and Hypothesis

Full-gradient SC-CBL produces a small positive aggregate validation result and
improves recall on both original-image folds, but AP/AP75 improves only on the
even fold and legacy micro/tiny AP regresses. Head-only distillation fails
outright. Together these observations imply:

1. high-resolution localization knowledge needs shared RoI representation
   adaptation; and
2. unrestricted distillation gradients sometimes conflict with the base
   detection/classification objective.

[PCGrad, NeurIPS 2020](https://arxiv.org/abs/2001.06782) projects one task
gradient away from another when their dot product is negative.
[CAGrad, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
instead optimizes average progress while regularizing for the worst local task
improvement. These are general multi-task optimizers; applying conflict control
specifically between detector supervision and cross-scale CBL distillation is
the project hypothesis, not yet a novelty claim.

## Proposed Objective

Keep the positive full-gradient SC-CBL forward path unchanged. Split loss into:

- `L_det`: classifier, base CBL, iterative CBL, RPN objectness, and RPN box
  losses; and
- `L_scale`: advantage-gated cross-scale CBL KL.

On shared RoI box-head parameters, compute gradients `g_det` and `g_scale`.
When `dot(g_det, g_scale) < 0`, project only the auxiliary gradient:

`g_scale_safe = g_scale - dot(g_scale, g_det) / ||g_det||^2 * g_det`.

Use `g_det + lambda * g_scale_safe` for shared parameters. Keep ordinary summed
gradients for the CBL distribution predictor and all non-shared detector
parameters. This asymmetry protects the base detector objective while still
allowing the shared localization representation to adapt, unlike head-only.

## Preregistered Gates

### Gate 0: gradient-conflict audit

Measure, without optimizer updates, at least 200 deterministic training
batches from seed 42:

- conflict rate `P(cos(g_det, g_scale) < 0)`;
- cosine mean, median, and quartiles;
- gradient norm ratio; and
- conflict rate by micro/tiny versus larger positive RoIs if separable without
  changing targets.

Continue only if conflict is material: at least 10% of audited batches have a
negative cosine and both gradient norms are finite/nonzero. Otherwise reject
the hypothesis without another validation run.

The audit state is frozen before execution: initialize the student exactly as
the SC-CBL local gate (`torchvision` Faster R-CNN ResNet50-FPN default weights,
random seed 42 CBL predictor, fixed `640/800` transform), use the fair20 EMA
epoch-5 `best.pt` teacher at `960/1200`, use the training weighted sampler and
copy-paste pipeline with batch size 4 and worker count 0, and perform no
optimizer update. Report the operational auxiliary gradient after its frozen
`0.25` weight and the unweighted norm ratio. Group audit RoIs at original-image
sqrt-area `<16 px` versus `>=16 px`; this grouping is diagnostic only and does
not change the objective. Compute the primary detector-versus-SC-CBL gradient
on all 200 batches and the two diagnostic size-band gradients every fifth
batch, yielding 40 preregistered band-audit batches while avoiding two extra
full shared-head gradient traversals on the other 160 batches.

### Gate 0 Result

Status: **failed**.

| Group | Valid batches | Conflict rate | Mean cosine | Median cosine | Mean weighted norm ratio |
|---|---:|---:|---:|---:|---:|
| All selected RoIs | 200 | 0.020 | 0.1497 | 0.1448 | 0.0612 |
| `<16 px` diagnostic | 40 | 0.000 | 0.1595 | 0.1551 | 0.0692 |
| `>=16 px` diagnostic | 36 | 0.000 | 0.0941 | 0.0798 | 0.0707 |

The unweighted auxiliary-to-detector norm ratio is approximately `0.2448` on
average because the operational audit includes the frozen `0.25` loss weight.
Across 200 batches there were `12,071` positive RoIs, of which `7,903` passed
the original teacher-IoU advantage gate. The four conflicting batches had
small negative cosines between `-0.0185` and `-0.0076`; projection would affect
too little of training to explain or reliably fix the fold inconsistency.

Artifact: `runs/ca_sc_cbl_gradient_conflict_audit_seed42.json`.

### Gate 1: technical optimizer test

Require exact synthetic projection tests, CUDA AMP compatibility, finite
four-step batch-size-4 optimization, no teacher gradient/state duplication,
and identical inference state/outputs when the conflict controller is absent.

### Gate 2: fresh-seed paired comparison

Do not shape another candidate on seed-42 validation. Run a fresh seed-123
two-epoch pair:

1. fixed-scale SA-ALW-ICBL baseline; and
2. CA-SC-CBL with the already frozen SC-CBL numeric configuration.

The method passes only if AP and AP75 improve, AR100 does not fall more than
`0.005`, and micro/tiny diagnostics do not both regress. Report both
original-image folds. No parameter sweep or seed-42 fallback is authorized.

### Cloud Decision

Only a fresh-seed pass can justify one full 20-epoch Kaggle run. Checkpoint
selection remains validation mAP50 `best.pt`; all artifacts require independent
reload. Do not access the existing locked test. Paper evidence must ultimately
include three seeds and an external public benchmark.

## Related Pages

- [[Cross-Scale CBL Localization Distillation Plan - 2026-08-01]]
- [[Coordinate-Reliable SC-CBL Plan - 2026-08-01]]
- [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]]
- [[Wiki Overview]]
- [[Wiki Log]]
