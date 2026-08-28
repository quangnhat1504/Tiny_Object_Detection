---
title: SAH-GD Advancement - 2026-06-02
type: analysis
created: 2026-06-02
updated: 2026-06-02
sources: [working/idea/5_sah_gd.md, wiki/sources/sah-gd-hybrid-metrics-comparison.md, working/code/9_hard_switch_p2.ipynb]
tags: [tiny-object-detection, sah-gd, metric, loss, research-direction]
---

## SAH-GD Advancement - 2026-06-02

## Question

Can the SAH-GD line of work progress further, or has metric-level innovation plateaued?

## What the ablation already told us

From [[SAH-GD Hybrid Metrics Comparison]], four variants landed within ~1% overall mAP
(0.567–0.577):

- **HARD_SWITCH won overall** (0.5770).
- **SCALE_TOPK won micro/tiny/AP@75/AR** (AP_micro 0.2947, AP@75 0.0453 — best strict
  localization across all runs).
- **SOFT_BLEND, the headline "differentiable transition", showed no advantage** over the hard
  switch and was the slowest.

Two honest implications:

1. The differentiable soft blend is **not** where the value is. More blending-shape variants
   are low-yield. The novelty should be reframed.
2. The component that actually moved micro metrics was **scale-adaptive positive sampling**
   (top-k by scale), i.e. an *assignment* idea, not a distance-formula idea.

So pure metric tweaking has largely plateaued (~1% spread). The remaining leverage is in the
two places the Gaussian metric is structurally weak, plus combining with architecture.

## The structural weakness to exploit: AP@75

Across every experiment AP@75 sits at 0.02–0.045. Root cause (verified in `9_hard_switch_p2`):
the box-regression loss is `L = 1 − exp(−β·D_H)` where `D_H` is the NWD/GCD Gaussian distance.
NWD/Wasserstein similarity is *deliberately* IoU-insensitive at small scale — good for stable
assignment, bad as the *sole* precision signal. For a 6px box, `D_N = W2 / C(s̄)` with
`C ≥ C_min = 4`, so a 1–2px center error already yields high similarity; the loss is "satisfied"
before the box is pixel-tight. The gradient `∂L/∂D_H = β·exp(−β·D_H)` also vanishes for far
predictions (hard cases get little signal) and is strongest near the target.

## Proposed advances (priority order)

### A. Dual-objective regression — the genuinely new, high-value lever (HIGH) — IMPLEMENTED

Decouple the two roles the metric is currently overloaded with:

```text
L_reg = (1 − S_H)               # coarse, stable, scale-adaptive (assignment-friendly)
      + γ · L_fine              # sharp localization signal at high overlap
```

where `L_fine` is DIoU/CIoU or a size-normalized L1 (`|Δ|/s̄`) that keeps strong gradient as
overlap → 1. This directly targets the universal AP@75 bottleneck while preserving SAH-GD's
micro stability. Motivation is clean and publishable: *"Gaussian metrics are IoU-insensitive by
construction; pair them with a precision term for strict localization on tiny objects."* Start
`γ ≈ 0.5–1.0`, ablate on AP@75 and AP_micro.

**Status (2026-06-02): implemented** in `working/code/10_dual_reg_p2.ipynb` (`HARD_SWITCH_P2_DUAL`),
built on the fair P2F config. `metric_fastrcnn_loss` now computes
`loss_box = (1 − S_H) + GAMMA_FINE · diou_loss(pred, tgt)` after the warmup epoch, with a
`USE_DUAL_REG` toggle and `GAMMA_FINE = 1.0`. The DIoU term gives non-zero gradient even for
non-overlapping tiny boxes (center penalty), so it complements rather than fights the Gaussian
similarity. Verified: identical boxes → 0, 2px shift → 0.27, no-overlap → 1.64, gradients finite.
**Result update (notebook 10, 2026-06-05):** the 16-epoch `HARD_SWITCH_P2_DUAL` run reached
`mAP(scale)=0.4516`, `AP_micro=0.2806`, `AP_tiny=0.4901`, `COCO AP@75=0.0129`, `AR@100=0.2886`,
and `72.54` detections/image. It reduced duplicate detections, but underperformed the earlier
P2 run and did not improve AP@75. Interpretation: applying `GAMMA_FINE=1.0` immediately after
warmup is too aggressive or poorly scheduled. Next sweep should use a smaller/ramped fine term
(`GAMMA_FINE=0.5`, delayed DIoU start), as reflected in `working/code/12_hard_switch_p2_topk_dual.ipynb`.

### B. SCALE_TOPK × P2 cross (HIGH, cheap — reuses winning pieces) — IMPLEMENTED

All SAH-GD ablations ran on the P3–P6 backbone (no P2). SCALE_TOPK won micro by giving micro GTs
more positives; P2 gives them resolution. The two should compound: more positives *and* finer
features for the same micro objects. This is the most promising untested combination and reuses
code that already exists. Run after the fair P2 baseline ([[P2 Experiment Result - 2026-06-02]]).

**Status (2026-06-02): implemented** in `working/code/11_scale_topk_p2.ipynb` (`SCALE_TOPK_P2`),
built on the fair P2F config. `hierarchical_label_assignment` now uses a per-GT scale-adaptive k
via `_gt_scale_topk`: `RFLA_K_MICRO=9` (s<8px), `RFLA_K_TINY=6` (8≤s<20px), `RFLA_K_OTHER=3`.
The earlier concern that large micro top-k adds noisy positives from a too-coarse level is
addressed precisely *because* P2 now supplies fine stride-4 anchors for the micro bin — this is
the synergy hypothesis. Metric (HARD_SWITCH) and regression (Gaussian similarity, non-dual) are
left unchanged to isolate the assignment effect.

**Result update (notebook 11, 2026-06-05):** the 16-epoch `SCALE_TOPK_P2` run reached
`mAP(scale)=0.4522`, `AP_micro=0.2821`, `AP_tiny=0.4971`, `COCO AP@75=0.0121`, `AR@100=0.2956`,
and `75.02` detections/image. It was slightly better than notebook 10 (`HARD_SWITCH_P2_DUAL`) on
overall AP, micro/tiny AP, AR, and speed, but still far below the earlier P2 result and did not
move AP@75. Interpretation: the large top-k schedule (`micro=9`, `tiny=6`) likely adds too many
noisy positives even with P2; the follow-up in `working/code/12_hard_switch_p2_topk_dual.ipynb`
therefore tempers top-k (`micro=6`, `tiny=5`) and delays/weakens the fine localization term.

### Combined TOPK + DUAL Follow-up (Notebook 12)

`working/code/12_hard_switch_p2_topk_dual.ipynb` combined the two follow-up ideas: scale-adaptive
top-k plus dual-objective regression. The first 16-epoch run (`HARD_SWITCH_P2_TOPK_DUAL`, before
the V2 config edits) reached `mAP(scale)=0.4724`, `AP_micro=0.3151`, `AP_tiny=0.5151`,
`COCO AP@75=0.0145`, `AR@100=0.2959`, and `70.45` detections/image.

This is the best of notebooks 10-12 and gives the cleanest detection count so far, but it still
does not recover the earlier P2 result (`mAP(scale)=0.5371`, `AP_micro=0.3586`) and leaves AP@75
very low. The V2 calibration has been split into `working/code/13_hard_switch_p2_topk_dual_v2.ipynb`
so it can be run without overwriting the notebook 12 result; treat it as a calibration attempt,
not as an already-validated improvement.

### C. FPN-normalized α_l — realize the "preferred form" never actually tested (MEDIUM)

The design doc's preferred gate is `α_l = sigmoid((s̄/stride_l − τ_f)/k)`, but the notebooks used
the pixel-level fallback because the RPN scaffold did not pass FPN strides into the metric. With
P2 now adding a genuine extra pyramid level, per-level stride normalization becomes meaningful:
the same pixel size means different "cells covered" on P2 vs P3. Implementing α_l makes the
transition resolution-aware and pairs naturally with the P2 work. Medium value because the
ablation suggests the *gate shape* matters less than assignment — but it is the principled form
and removes the resize-sensitivity limitation.

### Surgical RoI Localization Follow-up (Notebook 14)

After notebooks 10-12 underperformed, the next implementation step shifts away from TOPK/DUAL
and tests whether RoI feature resolution is the localization bottleneck. Implemented in
`working/code/14_p2_roialign14_convhead.ipynb`: starts from the clean P2F baseline, keeps
`RFLA_K=3`, keeps the HARD_SWITCH metric loss, disables scale-topk/DIoU, changes RoIAlign to
`14×14` with `sampling_ratio=4`, and replaces the plain MLP box head with a lightweight conv head
that learns a `14×14 → 7×7` downsample before the standard two-FC representation.

### D. Reframe the contribution (writing, not code)

Drop "differentiable soft blend" as the headline (it lost to hard switch). Lead with what the
data supports: (1) **adaptive normalization** `C(s̄) = sqrt((λs̄)² + C_min²)` for stable micro
gradients, and (2) **scale-adaptive positive sampling** `k(s)`. The hard switch is then the
simple, strong default; the soft blend is an ablation, not the claim.

### E. Density-aware weighting / metric Soft-NMS (LOW, deferred as designed)

det/img is still ~98; the deferred density weighting and duplicate regularizer could tighten
predictions and lift precision, but they add hyperparameters and are harder to ablate. Keep as
future work until A–C are settled.

## Bottom line

SAH-GD as a *blending formula* has plateaued. It can still advance meaningfully along three
lines: **(A)** dual-objective regression to finally move AP@75, **(B)** SCALE_TOPK combined with
P2 to compound the micro gains, and **(C)** the FPN-normalized gate that P2 now makes
worthwhile. The narrative should be reframed (D) around adaptive C(s) and scale-aware sampling
rather than the soft transition.

## Related Pages

- [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]]
- [[SAH-GD Hybrid Metrics Comparison]]
- [[P2 Experiment Result - 2026-06-02]]
- [[Tiny Object Architecture Improvement - 2026-05-31]]
- [[NWD]]
- [[GCD]]
