---
title: COCO Metrics Migration Plan - 2026-06-12
type: analysis
created: 2026-06-12
updated: 2026-06-12
sources: [paper/main.tex, wiki/sources/tiny-object-metrics-comparison-filled.md, eda/REPORT.md]
tags: [evaluation, coco-metrics, paper, migration, standardization]
---

# COCO Metrics Migration Plan - 2026-06-12

## Decision [2026-06-12]

**CONFIRMED:** Dùng đầy đủ **15 metrics** cho paper ALW:
- 9 COCO standard metrics (Main Table)
- 4 Maritime TOD-specific bins (Supplementary Table)
- 1 recall metric cho maritime bins
- 1 detection count (analysis)

Tất cả dùng **standard COCO IoU thresholds 0.50:0.05:0.95**. Custom IoU thresholds (0.25/0.25/0.35/0.50) bị **DROP hoàn toàn**.

---

## Danh sách đầy đủ 15 Metrics

### Main Table (Table III) — 9 metrics COCO Standard

| # | Metric | Định nghĩa | IoU Range | Area Filter | Table |
|---|--------|-----------|-----------|-------------|-------|
| 1 | **AP** | Average Precision | 0.50:0.05:0.95 | all areas | Main |
| 2 | **AP@50** | AP at IoU=0.50 | 0.50 | all areas | Main |
| 3 | **AP@75** | AP at IoU=0.75 (strict localization) | 0.75 | all areas | Main |
| 4 | **AP_S** | AP for small objects | 0.50:0.05:0.95 | area < 32² px² | Main |
| 5 | **AP_M** | AP for medium objects | 0.50:0.05:0.95 | 32² ≤ area < 96² | Main |
| 6 | **AP_L** | AP for large objects | 0.50:0.05:0.95 | area ≥ 96² | Main |
| 7 | **AR@100** | Average Recall (max 100 det/img) | 0.50:0.05:0.95 | all areas | Main |
| 8 | **AR_S** | AR for small objects | 0.50:0.05:0.95 | area < 32² | Main |
| 9 | **det/img** | Average detections per image | — | — | Main |

### Supplementary Table (Table IV) — 6 Maritime TOD Bins

| # | Metric | Định nghĩa (sqrt area) | IoU Range | Dataset Coverage | Table |
|---|--------|----------------------|-----------|-----------------|-------|
| 10 | **AP_vt** | Very tiny: 0–8 px | 0.50:0.05:0.95 | 27% (19,196 obj) | Supp |
| 11 | **AP_t** | Tiny: 8–16 px | 0.50:0.05:0.95 | 26% (18,040 obj) | Supp |
| 12 | **AP_s** | Small: 16–32 px | 0.50:0.05:0.95 | 39% (27,915 obj) | Supp |
| 13 | **AP_n** | Normal: >32 px | 0.50:0.05:0.95 | 8% (5,551 obj) | Supp |
| 14 | **AR_vt** | AR for very tiny | 0.50:0.05:0.95 | 27% | Supp |
| 15 | **AR@100** | Overall AR (repeat for context) | 0.50:0.05:0.95 | all | Supp |

**CRITICAL:** Tất cả 15 metrics dùng **CÙNG IoU matching standard 0.50:0.05:0.95** từ pycocotools.

---

## Mapping: OLD metrics → NEW metrics

| OLD (Custom) | Status | NEW (Standard) | Notes |
|-------------|--------|---------------|-------|
| mAP(scale) | ❌ **DROP** | AP (0.50:0.95) | Custom weighted aggregate → standard AP |
| mAP@50 | ✅ **KEEP** | AP@50 | Same metric, rename only |
| AP_micro (0-6px, IoU=0.25) | ⚠️ **REPLACE** | AP_vt (0-8px, IoU=0.50:0.95) | Bin widened to 8px, standard IoU |
| AP_tiny (6-16px, IoU=0.25) | ⚠️ **REPLACE** | AP_t (8-16px, IoU=0.50:0.95) | Bin adjusted, standard IoU |
| AP_small (16-64px, IoU=0.35) | ⚠️ **REPLACE** | AP_s (16-32px, IoU=0.50:0.95) | Bin narrowed to 32px, standard IoU |
| AP_large (64+px, IoU=0.50) | ⚠️ **REPLACE** | AP_n (32+px, IoU=0.50:0.95) | Bin starts at 32px, stricter IoU |
| COCO AP@50:75 | ⚠️ **REPLACE** | AP (0.50:0.95) | Widen IoU range to full COCO |
| COCO AP@75 | ✅ **KEEP** | AP@75 | Same metric |
| AR@100 | ✅ **KEEP** | AR@100 | Same metric |
| — (new) | ✅ **ADD** | AP_S, AP_M, AP_L | COCO size categories |
| — (new) | ✅ **ADD** | AR_S | COCO recall by size |
| — (new) | ✅ **ADD** | AR_vt | Maritime micro recall |

---

## Paper Table Layouts

### Table III: Main Comparison (COCO Standard)

```
| Method   |  AP  | AP@50 | AP@75 | AP_S | AP_M | AP_L | AR@100 | AR_S | det/img |
|----------|-----:|------:|------:|-----:|-----:|-----:|-------:|-----:|--------:|
| NWD      | X.XX | X.XX  | X.XX  | X.XX | X.XX | X.XX | X.XX   | X.XX | XXX     |
| GCD      | X.XX | X.XX  | X.XX  | X.XX | X.XX | X.XX | X.XX   | X.XX | XXX     |
| IGWD     | X.XX | X.XX  | X.XX  | X.XX | X.XX | X.XX | X.XX   | X.XX | XXX     |
| ALW      | X.XX | X.XX  | X.XX  | X.XX | X.XX | X.XX | X.XX   | X.XX | XXX     |
```

Caption: "Main comparison on SOD/TinyPerson-Sea (validation). Standard COCO evaluation with IoU thresholds 0.50:0.05:0.95. All methods share identical training harness; only the metric function differs. Bold = best, underline = second."

### Table IV: Maritime Size-Bin Analysis (Supplementary)

```
| Method   |  AP  | AP_vt | AP_t | AP_s | AP_n | AR_vt | AR@100 |
|----------|-----:|------:|-----:|-----:|-----:|------:|-------:|
| NWD      | X.XX | X.XX  | X.XX | X.XX | X.XX | X.XX  | X.XX   |
| GCD      | X.XX | X.XX  | X.XX | X.XX | X.XX | X.XX  | X.XX   |
| IGWD     | X.XX | X.XX  | X.XX | X.XX | X.XX | X.XX  | X.XX   |
| ALW      | X.XX | X.XX  | X.XX | X.XX | X.XX | X.XX  | X.XX   |
```

Caption: "Supplementary per-size analysis using maritime-relevant bins (sqrt area in px) with standard COCO IoU matching (0.50:0.05:0.95). Bins reflect maritime search-and-rescue detection ranges: vt=distant targets (0–8px, 27% of dataset), t=submerged swimmers (8–16px), s=partially visible persons (16–32px), n=nearby targets (>32px)."

---

## Vấn đề hiện tại (Legacy)

Paper ALW đang dùng **custom evaluation protocol** không standard:

### 1. Custom "mAP(scale)" metric — **DROPPED**
- ~~Instance-count-weighted aggregate với custom IoU thresholds~~
- **Thay bằng:** Standard COCO AP (0.50:0.05:0.95)

### 2. Custom COCO AP range — **FIXED**
- ~~COCO AP averaged over IoU 0.50:0.05:0.75 (6 thresholds)~~
- **Thay bằng:** Standard 0.50:0.05:0.95 (10 thresholds)

### 3. Tile-level evaluation — **MUST FIX**
- ~~`evaluate()` scores trên tiles~~
- **Thay bằng:** tile-inference → stitch → image-level NMS → vs original GT

---

## Implementation

### Code: COCO Standard Evaluation

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco_standard(gt_annotations, predictions, image_ids):
    """
    Standard COCO evaluation returning all 12 metrics.
    
    Returns:
        dict with keys: AP, AP50, AP75, APS, APM, APL,
                       AR1, AR10, AR100, ARS, ARM, ARL
    """
    coco_gt = COCO()
    coco_gt.dataset = {
        'images': [...],
        'annotations': gt_annotations,
        'categories': [...]
    }
    coco_gt.createIndex()
    
    coco_dt = coco_gt.loadRes(predictions)
    
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    stats = coco_eval.stats
    return {
        'AP': stats[0],       # AP @IoU=0.50:0.05:0.95
        'AP50': stats[1],     # AP @IoU=0.50
        'AP75': stats[2],     # AP @IoU=0.75
        'APS': stats[3],      # AP small (area < 32²)
        'APM': stats[4],      # AP medium (32² ≤ area < 96²)
        'APL': stats[5],      # AP large (area ≥ 96²)
        'AR1': stats[6],      # AR max 1 det/img
        'AR10': stats[7],     # AR max 10 det/img
        'AR100': stats[8],    # AR max 100 det/img
        'ARS': stats[9],      # AR small
        'ARM': stats[10],     # AR medium
        'ARL': stats[11],     # AR large
    }
```

### Code: Maritime TOD Bins

```python
import numpy as np

def evaluate_maritime_bins(coco_gt, coco_dt, image_ids):
    """
    Maritime-specific size bins using STANDARD COCO IoU matching.
    Bins defined by sqrt(area) in pixels.
    
    Returns:
        dict with AP_vt, AP_t, AP_s, AP_n, AR_vt, AR@100
    """
    bins = {
        'vt': (0, 64),         # very tiny: sqrt(area) 0-8px → area 0-64
        't':  (64, 256),       # tiny:      sqrt(area) 8-16px → area 64-256
        's':  (256, 1024),     # small:     sqrt(area) 16-32px → area 256-1024
        'n':  (1024, 1e10),    # normal:    sqrt(area) >32px → area >1024
    }
    
    results = {}
    for bin_name, (area_min, area_max) in bins.items():
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.params.areaRng = [[area_min, area_max]]
        coco_eval.params.areaRngLbl = [bin_name]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        results[f'AP_{bin_name}'] = coco_eval.stats[0]   # AP
        results[f'AR_{bin_name}'] = coco_eval.stats[8]    # AR@100
    
    return results
```

### Code: Image-Level Evaluation (Fix Tile Issue)

```python
import torch
from torchvision.ops import batched_nms

def inference_with_tiling(model, image, tile_size=512, overlap=64):
    """
    Tile inference → stitch → image-level NMS
    """
    tiles, offsets = tile_image(image, tile_size, overlap)
    
    all_boxes, all_scores, all_labels = [], [], []
    
    for tile, offset in zip(tiles, offsets):
        boxes, scores, labels = model(tile)
        boxes[:, [0, 2]] += offset[0]
        boxes[:, [1, 3]] += offset[1]
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)
    
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    
    keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
    return boxes[keep], scores[keep], labels[keep]
```

---

## Implementation Checklist

### Phase 1: Fix Evaluation Code
- [ ] Install `pycocotools`: `pip install pycocotools`
- [ ] Implement `evaluate_coco_standard()` → returns 12 COCO stats
- [ ] Implement `evaluate_maritime_bins()` → returns 6 maritime metrics
- [ ] Implement `inference_with_tiling()` with image-level NMS
- [ ] Unit test evaluation on 1-2 images
- [ ] Verify: tile-level vs image-level numbers documented

### Phase 2: Re-run ALL Experiments
- [ ] Re-run NWD with new COCO eval (byte-identical harness)
- [ ] Re-run GCD with new COCO eval
- [ ] Re-run IGWD with new COCO eval
- [ ] Re-run ALW with new COCO eval
- [ ] Collect all 15 metrics for each method
- [ ] Save results to new comparison table

### Phase 3: Update Paper
- [ ] Rewrite Table III (Main: 9 COCO metrics)
- [ ] Create Table IV (Supplementary: 6 maritime metrics)
- [ ] Rewrite Section 4.3 (Evaluation Metrics)
- [ ] Update Section 4.4 claims with new numbers
- [ ] Update abstract with new AP/AP@75 numbers
- [ ] Add COCO citation: Lin et al., ECCV 2014
- [ ] Verify all numbers consistent across abstract/tables/text

### Phase 4: Validation
- [ ] All methods use pycocotools.COCOeval
- [ ] Image-level evaluation confirmed (not tile-level)
- [ ] 15 metrics collected for all 4 methods
- [ ] Same inference pipeline for all methods
- [ ] Numbers in abstract match Table III exactly

---

## Expected Outcomes

### ⚠️ Results WILL change significantly:

1. **AP (0.50:0.95) sẽ THẤP HƠN AP@50:** ~0.3-0.5 × AP@50
2. **AP@75 có thể giảm:** Current 0.03-0.05 → possibly 0.01-0.03
3. **AP_S covers 92% dataset:** Most informative single metric
4. **AP_M/AP_L có rất ít samples:** High variance expected
5. **Rankings có thể thay đổi:** Focus on relative improvements

### ✅ Benefits:

1. Reviewers accept: COCO metrics = gold standard
2. Comparable to other papers
3. More rigorous evaluation
4. Better story: AP@75 improvement is genuine localization gain

---

## Paper Text Updates Required

### Section 4.3 — Evaluation Metrics (REWRITE):

> "We follow the standard COCO evaluation protocol~\cite{lin2014coco} with IoU thresholds 0.50:0.05:0.95. Table~\ref{tab:main} reports the primary metrics: AP (averaged over 10 IoU thresholds), AP@50, AP@75 for strict localization quality, AP\_S/AP\_M/AP\_L following COCO size categories (area $< 32^2$, $32^2$--$96^2$, $\geq 96^2$), AR@100 for recall, and AR\_S for small-object recall. Since 92\% of instances in our maritime dataset are smaller than $32\times32$ pixels, we additionally provide a supplementary per-size analysis (Table~\ref{tab:maritime}) using finer bins aligned to maritime SAR operational ranges: very-tiny (0--8\,px $\sqrt{\text{area}}$, 27\% of objects), tiny (8--16\,px), small (16--32\,px), and normal ($>$32\,px). These supplementary bins use the same COCO IoU matching (0.50:0.05:0.95) as the main evaluation. All inference is performed on full images via tiled prediction with image-level NMS, following standard practice."

---

## Related Pages

- [[Tiny Object Metrics Comparison Filled]]
- [[Tiny Object Detection Metrics]]
- [[ALW Paper Improvement Task List - 2026-06-10]]
- [[Anisotropic Log-Wasserstein Distance (ALW)]]
- [[ALW Full Paper Action Plan - 2026-06-10]]