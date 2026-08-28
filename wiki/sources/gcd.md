---
title: GCD
type: source
created: 2026-05-31
updated: 2026-05-31
sources: [GCD.pdf]
tags: [tiny-object-detection, metric, wasserstein, gcd]
---

## GCD

## Summary

`GCD.pdf` proposes Gaussian Combined Distance, a bounding-box metric designed to keep the smoothness advantage of Gaussian/Wasserstein metrics while adding scale invariance and more coupled optimization of center and shape terms.

## Key Claims

- Plain Wasserstein-style distance lacks scale invariance.
- NWD depends on a dataset-level constant `C`, which can make behavior inconsistent across object scales and datasets.
- GCD normalizes center and shape deviations by object dimensions and symmetrizes the formulation.
- The metric can be used for bounding-box regression and label assignment.

## Relevance To This Project

- The local experiment makes GCD the best current baseline overall:
  - `Best mAP(scale) = 0.5522`
  - `mAP@50 = 0.3483`
  - `AP_tiny = 0.5437`
  - `AP_small = 0.6370`
  - `AP_large = 0.7455`
- GCD also produces fewer detections than NWD/IGWD/ALW (`7,760`, `119.38` boxes/image), so it is the least noisy of the four tested runs.
- Its remaining weakness is localization quality: `COCO AP@75 = 0.0415`, still very low.

## Related Pages

- [[Tiny Object Detection Metrics]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
