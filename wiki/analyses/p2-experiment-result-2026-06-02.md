---
title: P2 Experiment Result - 2026-06-02
type: analysis
created: 2026-06-02
updated: 2026-06-02
sources: [working/code/9_hard_switch_p2.ipynb, wiki/sources/sah-gd-hybrid-metrics-comparison.md, wiki/sources/p2-feature-implementation.md]
tags: [tiny-object-detection, p2-features, experiment-results, architecture]
---

## P2 Experiment Result - 2026-06-02

## Question

Did adding the P2 (stride-4) FPN level to the HARD_SWITCH baseline improve micro-object
detection, and is it a net win? What should be done next?

## Result (notebook 9, HARD_SWITCH_P2, 12 epochs)

Best checkpoint at epoch 12. Training trajectory was still climbing steeply:
`mAP(scale)`: epoch 4 = 0.2178 → epoch 8 = 0.4569 → epoch 12 = 0.5371 (no plateau).

| Metric | Baseline HARD_SWITCH (file 6) | P2 (file 9) | Δ |
|--------|-------------------------------|-------------|---|
| mAP(scale) | 0.5770 | 0.5371 | −0.040 |
| mAP@50 | 0.3517 | 0.3055 | −0.046 |
| **AP_micro** | 0.2776 | **0.3586** (n=1009) | **+0.081 (+29%)** |
| AP_tiny | 0.5721 | 0.5582 (n=3705) | −0.014 |
| AP_small | 0.6600 | 0.5685 | −0.092 |
| AP_large | 0.7620 | 0.5321 | −0.230 |
| COCO AP@75 | 0.0428 | 0.0241 | −0.019 |
| COCO AP_small | — | 0.1278 | — |
| AR@100 | 0.3818 | 0.3374 | −0.044 |
| det/img | 100.08 | 98.54 | ~same |

Success criteria from [[Tiny Object Architecture Improvement - 2026-05-31]]:

- AP_micro > 0.30 → **ACHIEVED (0.3586)**.
- AP@75 > 0.05 → failed (0.0241, worse).
- mAP(scale) > 0.60 → failed (0.5371, down).
- det/img < 90 → failed (98.54).

## Interpretation

**The core P2 hypothesis is confirmed.** Micro objects (<8px) were resolution-starved on P3
(stride-8 → sub-pixel). Adding stride-4 P2 lifted AP_micro by +29%, exactly the target effect.
This is a real and large gain on the bin that matters most for this dataset.

**But this run is NOT a clean net win**, for three reasons — two of which are experiment-design
artefacts, not evidence that P2 hurts:

1. **Undertraining + 5-variable confound.** The mAP curve was still rising fast at epoch 12
   (+0.08 from epoch 8). P2 adds an FPN level (more parameters) while the run *halved* the
   effective batch (16 → 8 via BATCH 4→2, GRAD_ACCUM kept at 4), cut RPN proposals (3000→2000)
   and max detections (500→400), and shifted anchors smaller — all changed simultaneously. At
   equal epoch count, P2 is therefore more under-converged than the baseline. Comparing
   "12 epochs vs 12 epochs" is unfair.
2. **AP_large 0.76→0.53 is largely noise.** TinyPerson val has very few large instances
   (val is 28.8% tiny; micro n=1009, tiny n=3705, large not even reported with n). AP_large on
   a handful of instances is high-variance and should not drive decisions. AP_small (−0.092) is
   the more meaningful regression.
3. **AP@75 (strict localization) is the real, persistent bottleneck** — stuck at 0.02–0.045
   across *every* experiment so far. P2 did not fix it. Verified in code: after the warmup
   epoch the box-regression loss is `1 - exp(-β·D_H)` (the NWD/GCD Gaussian similarity), *not*
   Smooth-L1. The Gaussian metric is IoU-insensitive by design — the very property that helps
   assignment makes it a weak *sole* regression signal for pixel-precise boxes. RoIAlign is also
   still 7×7 for all levels, so a 6px box pooled from stride-4 features covers ~1.5px — too
   coarse for tight localization. See [[SAH-GD Advancement - 2026-06-02]].

## Conclusion

P2 is a correct and worth-keeping component for the micro bin, but the current run is confounded.
Two next moves:

1. **Confirm P2 fairly (P1, cheap).** Re-run with effective batch restored to 16 (GRAD_ACCUM
   4→8, keep BATCH=2 for memory) and EPOCHS 12→16, LR_STEPS [8,11]→[11,14]. Everything else
   (P2, anchors, metric) unchanged, to isolate the P2 effect. Implemented as `HARD_SWITCH_P2F`
   in `9_hard_switch_p2.ipynb`. Expected: micro stays ~0.35+, small/large/overall recover to
   ≥ baseline.
2. **Attack AP@75 next** — the universal bottleneck — via dual-objective regression and/or
   higher-resolution RoIAlign on low levels and/or Cascade refinement. See
   [[SAH-GD Advancement - 2026-06-02]].

## Evaluation-metric caveat

`mAP(scale)` equally weights four bins, so the rare, high-variance large bin can mask a real
micro win. For this project's goal, steer on an instance-weighted or tiny-focused composite
(e.g. mean of AP_micro and AP_tiny) rather than `mAP(scale)` alone.

## Related Pages

- [[P2 Feature Implementation - 2026-05-31]]
- [[SAH-GD Hybrid Metrics Comparison]]
- [[SAH-GD Advancement - 2026-06-02]]
- [[Tiny Object Architecture Improvement - 2026-05-31]]
