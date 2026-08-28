---
title: NWD
type: source
created: 2026-05-31
updated: 2026-05-31
sources: [NWD.pdf]
tags: [tiny-object-detection, metric, wasserstein, nwd]
---

## NWD

## Summary

`NWD.pdf` proposes Normalized Gaussian Wasserstein Distance for tiny object detection. The paper models each bounding box as a 2D Gaussian distribution and uses a normalized Wasserstein similarity to reduce the extreme sensitivity of IoU when tiny boxes shift by only one or a few pixels.

## Key Claims

- IoU-based assignment is unstable for tiny objects because a small absolute pixel shift can cause a large IoU drop.
- Bounding boxes can be represented as Gaussian distributions with center as mean and width/height as diagonal covariance terms.
- NWD can replace IoU in label assignment, NMS, and regression loss for anchor-based detectors.
- The normalization constant `C` should reflect the typical object size in the dataset.

## Relevance To This Project

- TinyPerson EDA shows `sqrt(area)` mean `15.7 px` and median `11.5 px`, so the project setting matches the motivation for NWD.
- In the local experiment, NWD is the best metric for `AP_micro` (`0.2652`) but has weak `mAP@50` (`0.2039`) and high detection count (`13,954`, `214.68` boxes/image).
- NWD is useful as a micro-object assignment candidate, but by itself it does not fix precision or localization.

## Related Pages

- [[Tiny Object Detection Metrics]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
