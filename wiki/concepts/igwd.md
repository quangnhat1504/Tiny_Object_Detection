---
title: IGWD
type: concept
created: 2026-05-09
updated: 2026-08-02
sources: [ALW.pdf, Improved_Gaussian_Wasserstein_Distance_A_Smooth_Adaptive_New_Metric_for_Remote_Sensing_Tiny_Object_Detection.pdf, tiny_object_metrics_comparison_filled.xlsx]
tags: [tiny-object-detection, metric, wasserstein]
---

## IGWD

## Definition

IGWD is a pair-area-normalized Gaussian-Wasserstein objective for tiny object
detection. The verified source is Hu, Chen, and Tang, IEEE TMM 2026, DOI
`10.1109/TMM.2026.3675527`.

For boxes `p,t`, it uses summed area `S=w_p h_p+w_t h_t`, distance
`sqrt(W2^2/S)`, and a fixed-beta exponential or rational similarity mapping.

## Issues Cited by the Source

- Shape is measured in Euclidean space rather than log-ratio space.
- A single isotropic denominator is used for both axes.

## Role in the Source

- The proposed ALW metric is framed as a principled replacement for IGWD.
- The source uses IGWD as the baseline for its ablation discussion.

## Local Experiment Note

- IGWD ranked second overall in the local comparison: `Best mAP(scale)=0.5187`, behind GCD `0.5522`.
- It was weaker than GCD on `AP_micro`, `AP_tiny`, `AP_small`, and `COCO AP@75`.
- The result suggests IGWD is a useful reference metric, but not the next primary baseline for this project.

## Related Pages

- [[Anisotropic Log-Wasserstein Distance (ALW)]]
- [[IGWD Paper]]
- [[Tiny Object Detection Metrics]]
- [[SA-ALW Paper Refinement Phase 0-2 - 2026-08-02]]

The formula and publication metadata are now verified against the accepted
manuscript; legacy local performance rows remain diagnostic only.
