---
title: IGWD Paper
type: source
created: 2026-05-31
updated: 2026-08-02
sources: [Improved_Gaussian_Wasserstein_Distance_A_Smooth_Adaptive_New_Metric_for_Remote_Sensing_Tiny_Object_Detection.pdf]
tags: [tiny-object-detection, metric, wasserstein, igwd]
---

## IGWD Paper

## Summary

The IGWD paper proposes Improved Gaussian Wasserstein Distance for remote-sensing tiny object detection. It targets limitations of IoU, DotD, NWD, and KLD by combining smooth geometric behavior with scale invariance and symmetry. The paper also proposes an Adaptive Channel Graph Convolution module, but this project only tested the metric side.

The source is now verified as Heng Hu, Si-Bao Chen, and Jin Tang, accepted in
IEEE Transactions on Multimedia (2026), DOI
`10.1109/TMM.2026.3675527`. The local PDF SHA-256 is
`7268ad1ad5fe5cab058138af8dfc4a081a621da6b0ee57fc82fed1a6b25186e1`.

## Key Claims

- IoU and KLD are too sensitive to small geometric deviations in tiny object detection.
- DotD and NWD are not scale-invariant.
- IGWD normalizes Wasserstein-style distance with area information to obtain scale invariance and symmetry.
- The metric can be embedded into loss, assignment, and NMS.

## Verified Formula

For boxes `p,t`, the paper defines `S(p,t)=w_p h_p+w_t h_t` and
`IGWD=sqrt(W2^2/S)`. It converts this distance to a `[0,1]` similarity using
either `exp(-beta * sqrt(IGWD))` or `1/(1+beta*IGWD)`, with fixed positive
`beta`. IGWD is pair-area normalized but isotropic; it does not contain
SA-ALW's target-scale beta or position-weight schedules.

## Relevance To This Project

- In the local experiment, IGWD is second-best overall:
  - `Best mAP(scale) = 0.5187`
  - `mAP@50 = 0.3294`
  - `AP_tiny = 0.5084`
  - `AP_small = 0.6156`
- It underperforms GCD on every main aggregate metric and has lower `AP_micro` (`0.1928`) than both NWD and GCD.
- `val_loss = 0.4121` suggests the tested configuration is less stable than GCD (`0.2197`).

## Related Pages

- [[IGWD]]
- [[Tiny Object Detection Metrics]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[SA-ALW Paper Refinement Phase 0-2 - 2026-08-02]]
