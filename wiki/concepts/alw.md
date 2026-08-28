---
title: Anisotropic Log-Wasserstein Distance (ALW)
type: concept
created: 2026-05-09
updated: 2026-08-02
sources: [ALW.pdf, tiny_object_metrics_comparison_filled.xlsx, paper-a-sa-alw-conference-refinement-plan]
tags: [tiny-object-detection, metric, wasserstein, log-ratio]
---

## Anisotropic Log-Wasserstein Distance (ALW)

## Definition

An ALW metric combines a Wasserstein-style position distance with log-ratio shape terms and anisotropic normalization for each axis. It is presented as a replacement for [[IGWD]] in tiny object detection.

## Structure

- Position terms are normalized separately for x and y.
- Shape terms use log ratios instead of Euclidean width and height differences.
- The resulting distance is converted into a similarity score with an exponential form.

## Claimed Properties

- Scale invariance
- Symmetry
- Dimensional consistency
- Numerical stability through clamping

For Paper A these properties apply only to the canonical base distance with
positive widths/heights before numerical stabilization. They do not transfer to
the target-conditioned SA-ALW schedules.

## Canonical Paper A Boundary (2026-08-02)

- Position denominators are per-axis mean-square scales.
- Shape terms are squared log ratios.
- Pure ALW excludes the legacy reliability gate and Charbonnier wrapper.
- Pairwise assignment and aligned regression are distinct tensor contracts.
- Historical `alw_full` checkpoints retain their legacy meaning and cannot be
  relabeled as canonical ALW evidence.
- Axis-specific normalization is not a standalone novelty claim: SimD already
  normalizes x/y terms separately, and GCD uses symmetric inverse-variance
  axis terms. The defensible ALW claim is the exact combination of per-axis
  mean-square center normalization and squared log-ratio shape.

See [[Paper A SA-ALW Conference Refinement Plan]] and
[[SA-ALW Paper Refinement Phase 0-2 - 2026-08-02]].

## Local Experiment Note

- ALW underperformed in the original local run: `Best mAP(scale)=0.3155`, `AP_tiny=0.3167`, `val_loss=0.5682`, `345.26` boxes/image.
- The improved local rerun in `working/code/4_alw.ipynb` performed worse: `Best mAP(scale)=0.1822`, `AP_micro=0.1029`, `AP_tiny=0.2190`, `COCO AP@75=0.0145`, and `599.20` boxes/image. The attempted dynamic top-k + reliability-gated robust shape merge did not transfer cleanly to this dataset.
- A separate reference run (`4_alw_rg_robust`) reached `mAP(scale)=0.5549`, `AP_micro=0.3096` on a different dataset — see [[RG-Robust ALW Implementation]]. Treat that result as non-comparable evidence that assignment can matter, not as a validated ALW improvement here.

## RG-Robust Improvements (2026-06-02)

The merged `working/code/4_alw.ipynb` adds two assignment-level improvements while keeping the original anisotropic position term and metric-NMS post-processing:

1. **Dynamic top-k by scale**: micro GTs get 6 positives, large get 3 — fixes positive-starvation for micro objects (main AP_micro driver).
2. **Reliability-gated robust shape**: GT-size gate `g(b)` lowers shape weight `λ(b)` and increases Charbonnier smoothing `ε(b)` for micro boxes, reducing gradient noise from noisy small-box size measurements.

See [[RG-Robust ALW Implementation]] for full details.

## Related Pages

- [[IGWD]]
- [[RG-Robust ALW Implementation]]
- [[Tiny Object Detection Metrics]]

The Paper A canonical formula is specified and tested independently in the
submission workspace; legacy performance claims remain diagnostic.
