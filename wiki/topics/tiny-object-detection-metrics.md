---
title: Tiny Object Detection Metrics
type: topic
created: 2026-05-09
updated: 2026-05-31
sources: [ALW.pdf, NWD.pdf, GCD.pdf, Improved_Gaussian_Wasserstein_Distance_A_Smooth_Adaptive_New_Metric_for_Remote_Sensing_Tiny_Object_Detection.pdf, RFLA.pdf, tiny_object_metrics_comparison_filled.xlsx, eda/REPORT.md]
tags: [tiny-object-detection, metrics, loss-functions]
---

## Tiny Object Detection Metrics

## Focus

This topic covers bounding-box metrics, label assignment, loss functions, and post-processing for tiny object detection, especially Gaussian/Wasserstein-style objectives.

## What The Sources Add

- [[NWD]] argues that IoU is too sensitive to one-pixel shifts on tiny boxes and replaces IoU with normalized Gaussian Wasserstein similarity.
- [[GCD]] argues that Wasserstein-style metrics need scale invariance and coupled center/shape optimization.
- [[IGWD Paper]] normalizes Wasserstein geometry with area information to improve smoothness, symmetry, and scale invariance.
- [[Anisotropic Log-Wasserstein Distance (ALW)]] argues that tiny-object shape should be measured in log-ratio space and that x/y position terms need anisotropic normalization.
- [[RFLA]] shifts attention from box/point priors to Gaussian receptive-field based label assignment.
- [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]] proposes a soft scale-adaptive blend of adaptive NWD and GCD.

## Local Experiment Result

From [[Tiny Object Metrics Comparison Filled]]:

| Metric | Best mAP(scale) | mAP@50 | AP_micro | AP_tiny | AP_small | COCO AP@75 | Avg det/img |
|--------|----------------:|-------:|---------:|--------:|---------:|-----------:|------------:|
| NWD | 0.3967 | 0.2039 | **0.2652** | 0.4741 | 0.3603 | 0.0376 | 214.68 |
| GCD | **0.5522** | **0.3483** | 0.2582 | **0.5437** | **0.6370** | **0.0415** | **119.38** |
| IGWD | 0.5187 | 0.3294 | 0.1928 | 0.5084 | 0.6156 | 0.0312 | 133.92 |
| ALW | 0.1822 | 0.1256 | 0.1029 | 0.2190 | 0.1656 | 0.0145 | 599.20 |

## Practical Implication

- GCD is the strongest current baseline and should be the default for the next run.
- NWD is still valuable as a micro-object ablation because it has the highest `AP_micro`.
- All four runs have weak strict localization (`COCO AP@75 <= 0.0415`), so metric replacement is not enough.
- The next improvement should target high-resolution feature coverage, anchor/proposal sizing, and NMS/post-processing calibration.
- SAH-GD should be treated as an ablation candidate, not a replacement for architecture fixes, until it beats adaptive NWD and hard-switch baselines.

## Current Ablation Direction

The active Kaggle ablation plan tests:

| Ablation | Purpose |
|----------|---------|
| Adaptive NWD only | Tests whether adaptive `C(s)` is already enough. |
| Hard switch NWD/GCD | Tests whether a simple non-smooth rule beats soft blending. |
| SAH-GD soft blend | Tests the main proposed differentiable transition. |
| SAH-GD + scale top-k | Tests whether micro GT needs more positives. |

## Related Pages

- [[Anisotropic Log-Wasserstein Distance (ALW)]]
- [[IGWD]]
- [[IGWD Paper]]
- [[NWD]]
- [[GCD]]
- [[RFLA]]
- [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[Tiny Object Metric Ablation Plan - 2026-05-31]]
