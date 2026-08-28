---
title: Phase 0 Dataset Statistics - 2026-07-01
type: analysis
created: 2026-07-01
updated: 2026-07-01
sources: [eda/instances.csv, eda/summary.json, eda/Phase0_report.md, eda/REPORT.md]
tags: [tiny-object-detection, phase-0, statistics, dataset-analysis]
---

## Phase 0 Dataset Statistics - 2026-07-01

## Question

What are the key data characteristics for SOD-TinyPeopleInSea that inform the SA-ALW / Cascaded Routing design?

## Key Findings

### Scale Percentiles (sqrt-area)
- **P10=5.6, P90=28.7** — used as s_min/s_max for SA-β and SA-position-weight.
- dry-person and wet-swimmer have similar distributions — no need for separate class parameters.

### H0.1 Rejected
Wet-swimmer is NOT smaller than dry-person (P50: 11.92 vs 11.30). The hypothesis was wrong — use shared s_min/s_max for both classes.

### Aspect Ratio — Dry-person is the elongated class
- dry-person: 43.6% tall (aspect<0.5) — standing people
- wet-swimmer: 10.2% tall, 4.0% flat — more square
- Anisotropic normalization helps dry-person more than wet-swimmer

### Annotation Noise: Low (1.6%)
Only 1,113 boxes (1.6%) have w<3 or h<3 — no special cleaning needed.

### Split Balance
Valid set has slightly larger objects (P50=13.3 vs train 11.5). Test set has smaller objects (P50=10.7). Minor concern — note when comparing validation vs test results.

### Patch Data Generated
10,717 patches (9,863 train + 854 valid) from context-ratio 1.5 cropping. Ready for Phase 1.3 patch-based Faster R-CNN baseline.

## Related Pages
- [[P2 Experiment Result - 2026-06-02]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[SAH-GD Advancement - 2026-06-02]]
