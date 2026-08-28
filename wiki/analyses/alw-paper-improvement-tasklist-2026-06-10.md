---
title: ALW Paper Improvement Task List - 2026-06-10
type: analysis
created: 2026-06-10
updated: 2026-06-10
sources: [paper/main.tex, paper/README.md, tod-alw.ipynb, wiki/sources/tiny-object-metrics-comparison-filled.md, eda/REPORT.md]
tags: [tiny-object-detection, alw, paper, task-list, action-plan]
---

## ALW Paper Improvement Task List - 2026-06-10

## Question

The ALW paper draft (`paper/main.tex`, IEEEtran) is written from a single run of
`tod-alw.ipynb` plus the local metric-comparison table. What must be done to make it
submission-grade, ordered from most severe to normal?

## Context (current draft state)

- **Proposed method:** ALW (anisotropic per-axis position normalization + log-ratio shape),
  used in RFLA label assignment + RoI box-regression loss. Plain Faster R-CNN + ResNet-50-FPN.
- **Baselines in main table:** NWD/GCD/IGWD (`†` = from [[Tiny Object Metrics Comparison Filled]],
  NOT byte-identical re-runs). Only IGWD→ALW is a controlled swap.
- **Headline numbers (best epoch 8):** mAP(scale)=0.5439, mAP@50=0.3435, AP_micro=0.2495,
  AP_tiny=0.5283, AP_small=0.6325, AP_large=0.7956, COCO AP@50:75=0.1893, **AP@75=0.0536**,
  AR@100=0.4003. ALW beats IGWD on every metric (AP@75 +72% rel) but trails GCD on coarse
  mAP@50/mAP(scale)/micro/tiny/small.
- **Scope lock:** exclude SAC / HFP / SDP / P2 / SAH-GD and all "future" improvements — see
  scope memory and [[Tiny Object Detection Metrics]].

## 🔴 CRITICAL — claims invalid / likely reject without these

1. **Re-run NWD / GCD / IGWD on a byte-identical harness.** Today only IGWD→ALW is controlled;
   the NWD/GCD rows are from the old comparison table. The whole main table is contestable until
   all four metrics share code/seed/schedule, swapping only `compute_rfd` + box loss.
2. **Multiple seeds, report mean±std.** Single seed only. TOD variance is high; the +2.5 pt
   mAP(scale) gain over IGWD may be noise. Run ≥3 seeds for ALW + baselines; if the gap is
   within std, reframe the headline.
3. **Validate on a public benchmark (AI-TOD / AI-TOD-v2).** One private maritime dataset is not
   acceptable to CV reviewers. Need a recognized benchmark for comparability and to rule out
   single-distribution overfit. Longest pole — start first.
4. **Fix / unify the evaluation protocol (tile-level vs image-level).** `evaluate()` scores on
   tiles, so per-bin `n_gt` is inflated by tile overlap and COCO AP is tile-level — non-standard.
   Standard practice scores on full images (tile-inference → stitch → NMS → vs original GT).
   Pick one protocol and apply it to every method.

## 🟠 HIGH — needed for a convincing contribution

5. **Per-component ablation.** Core thesis = "anisotropy + log-ratio shape", but IGWD→ALW only
   measures the *combined* effect (and IGWD differs in more than two ways). Run the 4-cell grid:
   (isotropic+Euclid) → (anisotropic+Euclid) → (isotropic+log) → (anisotropic+log = ALW).
6. **Nail the GCD relationship.** ALW trails GCD on mAP(scale), mAP@50, AP_micro/tiny/small;
   wins only AP@75 / AR / large. Either improve ALW to win ≥1 coarse axis, or reframe the
   contribution explicitly as "best strict localization" rather than implying overall SOTA.
7. **Verify the reported run is clean ALW.** Notebook has dead-code branches
   (`ALW_SHAPE_RELIABILITY_THR`, `ALW_IGWD_ALPHA_MAX` hybrid, `ALW_SOFT_LOG_SIZE_THR`). Log
   suggests none were active, but confirm and freeze/remove them for reproducibility.
8. **β sensitivity + gradient analysis.** β=8 is hard-coded with no justification — sweep
   {4, 8, 16}. Add a gradient analysis (as GCD does) showing the loss is scale-balanced, ideally
   with a small synthetic tiny-vs-large box experiment at equal relative error.

## 🟡 MEDIUM — quality and credibility

9. **More baselines:** Smooth-L1/IoU, DotD, GWD, CIoU — position ALW in the full landscape, not
   just the Gaussian family.
10. **Figures** (draft has none): geometry diagram (isotropic vs anisotropic; log-ratio
    intuition), training-curve chart (data in `tab:curve`), qualitative ALW vs IGWD on tiny
    objects.
11. **Post-processing / over-detection.** Inference emits ~909 boxes/img → precision/duplicate
    problem; AP@75 low everywhere. Discuss + try ALW-NMS (currently future work).

## 🟢 NORMAL — pre-camera-ready polish

12. Verify the IGWD citation (`igwd2024` is a placeholder); add real venue/authors.
13. Describe the dataset properly (source, license, split rationale, why a private maritime SAR set).
14. Notation table; complexity/runtime of ALW vs IoU.
15. Reproducibility: notebook → script + config, publish seeds, organize code release.
16. Writing polish; check number consistency across abstract / tables / text.

## Suggested execution order

Start #3 (AI-TOD, slow) in parallel with #1, #2 (re-runs + seeds) — these are blockers and
compute-heavy. While runs are in flight, do #5, #7, #8 (ablation/verify, cheap). Figures and
polish (#10–16) last.

## Related Pages

- [[Anisotropic Log-Wasserstein Distance (ALW)]]
- [[Tiny Object Metrics Comparison Filled]]
- [[Tiny Object Detection Metrics]]
- [[IGWD Paper]]
