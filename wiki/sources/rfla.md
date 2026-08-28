---
title: RFLA
type: source
created: 2026-05-31
updated: 2026-05-31
sources: [RFLA.pdf]
tags: [tiny-object-detection, label-assignment, receptive-field, rfla]
---

## RFLA

## Summary

`RFLA.pdf` proposes Gaussian Receptive Field based Label Assignment for tiny object detection. Instead of relying on box priors or point priors, RFLA models the feature receptive field as a Gaussian prior and measures similarity between receptive fields and ground-truth boxes.

## Key Claims

- Standard anchor-based box priors and anchor-free point priors are suboptimal for tiny objects.
- Tiny ground-truth samples can become outliers under IoU-threshold assignment or center sampling.
- Receptive Field Distance (RFD) better represents whether a feature location can learn a tiny object.
- Hierarchical Label Assignment (HLA) balances learning across tiny objects.

## Relevance To This Project

- The local experiments were run on a Faster R-CNN + RFLA style setup, so RFLA is the assignment backbone around which the metric variants were tested.
- Because all tested metrics still have very low `COCO AP@75`, the next improvement should inspect assignment density, anchor/stride coverage, and duplicate suppression rather than only swapping the box metric.
- RFLA supports the direction of making label assignment more receptive-field-aware for micro objects.

## Related Pages

- [[Tiny Object Detection Metrics]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
