---
title: Cascaded Routing Implementation Plan - 2026-07-01
type: analysis
created: 2026-07-01
updated: 2026-07-01
sources: [raw/plan.md, raw/detail_implement.md, common/metrics/sa_alw.py, scripts/train_yolo.py, scripts/generate_patches.py]
tags: [tiny-object-detection, implementation-plan, cascade, routing]
---

## Cascaded Routing Implementation Plan - 2026-07-01

## Question

How to implement, ablate, and validate a cascaded YOLO + Faster R-CNN pipeline with SA-ALW metric on SOD-TinyPeopleInSea?

## Execution Plan (6 Phases)

### Phase 0 — Data Preparation ✅ COMPLETED

See [[Phase 0 Dataset Statistics - 2026-07-01]].

- Dataset statistics complete (percentiles, aspect ratios, annotation quality)
- s_min=5.6, s_max=28.7 from P10/P90
- 10,717 patches generated in `data/patches/`
- H0.1 rejected (wet-swimmer NOT smaller)

### Phase 1 — Independent Baselines (before hybrid comparison)

Three baselines required:

| # | Baseline | Code | Metric |
|---|----------|------|--------|
| 1.1 | YOLOv8n/v11n full-image | `scripts/train_yolo.py` | mAP@50, FPS, Precision, Recall |
| 1.2 | Faster R-CNN (ResNet50-FPN) full-image | `common/` — ciou everywhere | Scale AP, COCO mAP |
| 1.3 | Faster R-CNN on patches | `common/` — ciou everywhere + `use_patches=True` | Same as 1.2 |

All use the same annotation/data to ensure fair comparison. Metrics: mAP@50, mAP@50:95, AP_micro/tiny/small/large, FPS, Precision, Recall.

### Phase 2 — Metric Chain Ablation (on Faster R-CNN full-image)

Test metrics in isolation before adding Router complexity:

| Step | Config | Purpose |
|:----:|--------|---------|
| 2.1 | IoU (MaxIoUAssigner + SmoothL1) | Baseline (from Phase 1) |
| 2.2 | NWD | Wasserstein distance |
| 2.3 | IGWD (β=8 fixed) | Scale-invariant Wasserstein |
| 2.4 | IGWD + log-shape only | Ablate ALW's log-ratio |
| 2.5 | IGWD + anisotropic-S only | Ablate ALW's anisotropy |
| 2.6 | ALW full (log-shape + anisotropic, β fixed) | Full ALW |
| 2.7 | SA-ALW (SA-β only, w_pos=1) | Scale-Adaptive beta |
| 2.8 | SA-ALW full (SA-β + SA-pos-weight) | Complete SA-ALW |

Key Hypothesis H2.1-H2.5 tracked — see full list below.

### Phase 3 — SAALWAssigner (RPN Integration)

- Implement `SAALWAssigner` based on SA-ALW similarity
- Grid search `pos_sim_thr` × `neg_sim_thr` on validation (recall of RPN proposals)
- Ablate `topk_fallback` ∈ {3, 6, 9, 12} — hypothesis H3.1 (non-monotonic optimal)
- Compare with ATSS assigner (baseline) — hypothesis H3.2

### Phase 4 — Uncertainty Router + Cascaded Pipeline

- Implement `UncertaintyRouter` with 3 routing criteria
- Grid search `conf_low` × `conf_high` × `area_thr`
- Pareto curve: recall vs compute-cost
- **Critical H4.1**: measure YOLO's complete false-negative rate — if >5-10%, add blind-spot scanning
- Compare 3 patch strategies: fixed-size, context-ratio dynamic, sliding-window (H4.2)

### Phase 5 — Fusion & NMS

- Weighted Boxes Fusion (tune YOLO vs RCNN weights)
- ALW-based Soft-NMS (replace IoU with 1 - ALW_sim)
- Hypothesis H5.1: ALW-NMS handles nearby tiny objects better

### Phase 6 — Full System Evaluation

- Final comparison table: 3 baselines + full hybrid system
- FPS measured as distribution (not mean — depends on route ratio)
- Locked test set (no leakage from Phase 2-5 grid search)
- Error taxonomy: YOLO miss vs Router miss vs RCNN miss vs Fusion wrong

## All Hypotheses

| ID | Hypothesis | Test |
|:--:|-----------|------|
| H0.1 | ~~wet-swimmer smaller than dry-person~~ | **REJECTED** — opposite true |
| H2.1 | IGWD > NWD on AP_vt, small gain on mAP | Phase 2.2 vs 2.3 |
| H2.2 | ALW > IGWD stronger on dry-person (elongated) | Phase 2.6, class-separated AP |
| H2.3 | SA-β and SA-pos-weight have overlapping contribution (non-additive) | Phase 2.7 vs 2.8, bootstrap CI |
| H2.4 | log_clamp affects training stability | Test {1.5, 2.0, 3.0, 5.0, no clamp} |
| H2.5 | Rational normalization vs exp normalization | Compare `1/(1+β·d)` vs `exp(-β·d)` |
| H3.1 | topk_fallback has non-monotonic optimum (≈6-9) | Grid {3, 6, 9, 12} |
| H3.2 | SAALWAssigner > ATSS on AP_vt but ≤ ATSS on general AP | Direct comparison |
| H4.1 | Cascade blind spot when YOLO completely misses | Measure FN rate of YOLO alone |
| H4.2 | Sliding-window > context-ratio for recall but higher compute | Pareto curve |
| H5.1 | ALW-NMS better than IoU-NMS for nearby tiny objects | Dedicated test cases |

## Code Structure

```
common/
├── config.py              # All hyperparams + paths
├── dataset.py             # YOLOTinyDataset + build_training_datasets()
├── model.py               # Faster R-CNN builder (4 placements)
├── train_utils.py         # Training loop + EMA + scheduler
├── eval_utils.py          # FPS, Precision, Recall, Scale AP, COCO mAP
├── generate_notebooks.py  # Experiment notebook generator
├── metrics/
│   ├── iou.py             # CIoU baseline
│   ├── nwd.py             # Normalized Wasserstein Distance
│   ├── igwd.py            # Improved Gaussian Wasserstein Distance
│   ├── alw.py             # ALW (proposed, 4 ablations)
│   └── sa_alw.py          # SA-ALW (scale-adaptive beta + pos-weight) [NEW]
scripts/
├── train_yolo.py          # YOLO baseline training [NEW]
└── generate_patches.py    # Patch data generator [NEW]
```

## Related Pages

- [[Phase 0 Dataset Statistics - 2026-07-01]]
- [[Cascaded Uncertainty Routing]]
- [[Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)]]
- [[SAH-GD Advancement - 2026-06-02]]
