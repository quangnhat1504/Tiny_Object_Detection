---
title: Tiny Object Metrics Comparison Filled
type: source
created: 2026-05-31
updated: 2026-06-05
sources: [tiny_object_metrics_comparison_filled.xlsx]
tags: [tiny-object-detection, experiment-results, metrics]
---

## Tiny Object Metrics Comparison Filled

## Summary

`tiny_object_metrics_comparison_filled.xlsx` contains the filled comparison table for four local metric experiments: NWD, GCD, IGWD, and ALW.

## Main Result Table

| Metric | Best mAP(scale) | mAP@50 | val_loss | AP_micro | AP_tiny | AP_small | AP_large | COCO AP@50:75 | COCO AP@75 | AR@100 | Avg det/img |
|--------|----------------:|-------:|---------:|---------:|--------:|---------:|---------:|--------------:|-----------:|-------:|------------:|
| NWD | 0.3967 | 0.2039 | 0.2655 | **0.2652** | 0.4741 | 0.3603 | 0.2597 | 0.1162 | 0.0376 | 0.3852 | 214.68 |
| GCD | **0.5522** | **0.3483** | **0.2197** | 0.2582 | **0.5437** | **0.6370** | **0.7455** | **0.1821** | **0.0415** | **0.3939** | **119.38** |
| IGWD | 0.5187 | 0.3294 | 0.4121 | 0.1928 | 0.5084 | 0.6156 | 0.7075 | 0.1639 | 0.0312 | 0.3650 | 133.92 |
| ALW | 0.1822 | 0.1256 | 0.6340 | 0.1029 | 0.2190 | 0.1656 | 0.1744 | 0.0652 | 0.0145 | 0.3589 | 599.20 |

## Direct Conclusions

- GCD is the best overall metric in this local run.
- NWD is narrowly best on `AP_micro`, which matters because the EDA reports `19,196` objects in the `(0,8)` px bin.
- ALW is the weakest tested configuration after the improved `4_alw.ipynb` rerun: lowest `Best mAP(scale)`, highest `val_loss`, slowest inference, and most detections.
- `COCO AP@75` is poor for all metrics, so localization remains the dominant bottleneck.
- Detection counts are high, especially ALW and NWD, so post-processing and duplicate suppression need attention.

## Related Pages

- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[Tiny Object Detection Metrics]]
