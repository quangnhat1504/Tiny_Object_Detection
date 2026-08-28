---
title: ALW
type: source
created: 2026-05-09
updated: 2026-05-31
sources: [ALW.pdf]
tags: [tiny-object-detection, metric, loss, wasserstein]
---

## ALW

## Summary

This source proposes Anisotropic Log-Wasserstein Distance (ALW), a metric for tiny object detection that aims to fix two structural issues in IGWD: shape measured in Euclidean space instead of log-ratio space, and isotropic normalization that uses the same denominator for both axes.

## Key Claims

- Replace the IGWD shape terms with log-ratio terms such as `ln(wp / wt)` and `ln(hp / ht)`.
- Replace isotropic position normalization with anisotropic per-axis denominators.
- Preserve the Wasserstein-style position structure while improving scale invariance and dimensional consistency.
- Use a similarity form `ALW_sim = exp(-beta * sqrt(ALW2))` and a loss `CALW = 1 - ALW_sim`.

## Formal Properties Claimed

- Dimensional consistency: all four terms in `ALW2` are dimensionless.
- Scale invariance: scaling both boxes by the same factor preserves the metric.
- Symmetry: swapping prediction and target does not change the distance.
- Numerical stability: widths and heights are clamped before taking logs.

## Experimental Direction

- The source proposes ablations on AI-TOD to separate the contribution of the log-shape replacement and the anisotropic normalization.

## Local Experiment Note

- In the local four-metric comparison, ALW was the weakest tested configuration. The original run reached `Best mAP(scale)=0.3155`, `val_loss=0.5682`, and `345.26` detections/image.
- The improved `working/code/4_alw.ipynb` rerun underperformed further: `Best mAP(scale)=0.1822`, `AP_micro=0.1029`, `AP_tiny=0.2190`, `COCO AP@75=0.0145`, and `599.20` detections/image.
- This does not invalidate the geometric idea, but it means the current implementation/configuration should not be the next baseline.
- If revisited, ALW should first be checked for log/clamp stability, beta sensitivity, assignment quality, and especially NMS calibration.

## Implementation Note

- The source includes a PyTorch-style loss implementation for `ALW`.

## Caveat

❓ Unverified: this source was OCR-extracted from a PDF, so some equations and symbols may need manual correction against the original image.
