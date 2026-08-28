---
title: Tiny Object Metric Experiment - 2026-05-31
type: analysis
created: 2026-05-31
updated: 2026-05-31
sources: [tiny_object_metrics_comparison_filled.xlsx, NWD.pdf, GCD.pdf, Improved_Gaussian_Wasserstein_Distance_A_Smooth_Adaptive_New_Metric_for_Remote_Sensing_Tiny_Object_Detection.pdf, ALW.pdf, RFLA.pdf, eda/REPORT.md]
tags: [tiny-object-detection, experiment-results, improvement-plan]
---

## Tiny Object Metric Experiment - 2026-05-31

## Question

After testing NWD, GCD, IGWD, and ALW on the same Faster R-CNN + RFLA style setup, what is the most useful next improvement step?

## Dataset Facts From EDA

- `70,702` objects across `1,570` images.
- `79.5%` of objects have `sqrt(area) < 20 px`.
- `92.1%` of objects have `sqrt(area) < 32 px`.
- The `(0,8)` px bin alone has `19,196` instances.
- Average density is `45.03` objects/image, with a maximum of `730`.
- Class imbalance is about `1.9:1` in favor of `dry-person`.

## Experiment Findings

| Finding | Evidence |
|---------|----------|
| Best overall metric | GCD: `Best mAP(scale)=0.5522`, `mAP@50=0.3483`, `AP_tiny=0.5437`, `AP_small=0.6370` |
| Best micro-object metric | NWD: `AP_micro=0.2652`, slightly above GCD `0.2582` |
| Best balance/noise tradeoff | GCD: strongest AP and lowest average detections/image among tested metrics |
| Weakest tested setup | ALW improved rerun: `mAP(scale)=0.1822`, `val_loss=0.6340`, `599.20` detections/image |
| Shared bottleneck | All metrics have very low `COCO AP@75` (`0.0312` to `0.0415`) |

## Interpretation

Metric replacement helped, but it did not solve the core failure mode. GCD gives the best assignment/loss behavior in the current setup, while NWD remains useful for the smallest objects. The shared low `AP@75` means the model is often close enough for loose matching but not accurate enough for stricter localization. The high detection counts also point to duplicate boxes and insufficient post-processing calibration in dense scenes.

## Recommended Next Improvement

Use **GCD as the main baseline**, then improve the pipeline around micro-object localization:

1. Add or enable a high-resolution detection level such as `P2` / stride-4 features.
2. Retune anchors or proposal sizes around the EDA bins: `(0,8)`, `[8,12)`, `[12,20)`, `[20,32)`.
3. Calibrate NMS/post-processing from the GCD run first, because it already has the best AP and lowest duplicate rate.
4. Optionally use NWD only as a micro-object assignment branch or ablation, because it is best on `AP_micro` but weaker globally.

## Why This Step

The data distribution says the model must learn objects around `2-20 px`, while the result table says strict localization is failing for every metric. A new metric alone is unlikely to fix that. The most defensible next step is to keep the strongest metric baseline, GCD, and improve feature resolution plus proposal/anchor coverage for the micro and tiny bins.

## Related Pages

- [[Tiny Object Metrics Comparison Filled]]
- [[Tiny Object Detection Metrics]]
- [[GCD]]
- [[NWD]]
- [[RFLA]]
