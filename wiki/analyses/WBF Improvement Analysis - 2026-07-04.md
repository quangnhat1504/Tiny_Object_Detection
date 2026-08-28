---
title: WBF Improvement — Root Cause Analysis & Plan
type: analysis
created: 2026-07-04
updated: 2026-07-04
sources: [scripts/tune_cascade.py, common/cascade.py, common/eval_utils.py]
tags: [wbf, cascade, phase4, improvement-plan]
---

## WBF Improvement — Root Cause Analysis & Plan

### Problem

Grid search Phase 4: WBF (iou_thr=0.55, score_thr=0.20) đạt mAP(scale)=0.6017,
nhưng **AP_small (-21.4%)** và **AP_large (-19.8%)** tụt mạnh so với
standalone FRCNN. Trong khi AP_micro (+14.8%) và AP_tiny (+5.9%) tăng.

### Root Cause Analysis

#### 1. Metric comparison was unfair

- **Standalone eval (eval_utils.py)**: đánh giá **per-tile**. GT box cũng bị cắt
  theo tile boundary (dataset.py `_clip_boxes`). Một object 200px nằm trên 2 tile
  → tạo ra 2 GT nhỏ hơn → mAP bị inflate vì model chỉ cần detect partial box
  trên mỗi tile.

- **Cascade eval (eval_cascade.py)**: remap prediction về full-image, WBF merge,
  so sánh với **full-image GT** (không cắt tile). Đây mới là metric thực tế.

→ Chênh lệch giữa standalone và cascade không đơn thuần là WBF làm xấu, mà
một phần do **cách đánh giá khác nhau**. Standalone "dễ" hơn vì GT cũng bị cắt.

#### 2. WBF average shrinks large objects

```
Object 200×100px nằm trên 3 tile (512×512, stride 448):
  Tile A: thấy object từ x=500-512 (12px rộng) → predict box ~(500, 195, 515, 300)
  Tile B: thấy object từ x=448-700 (toàn bộ)   → predict box ~(500, 200, 700, 295)
  Tile C: không thấy (object nằm ngoài)

WBF average → fused box = mean([(500,195,515,300), (500,200,700,295)])
                         ≈ (500, 197, 607, 297)
                         width = 107px (vs GT 200px) → SHRUNK 46%
```

Nguyên nhân: WBF clustering dùng IoU threshold cố định. Hai partial prediction
của cùng một object có IoU thấp (0.08) → không merge được → mỗi partial box
thành cluster riêng → kết quả: nhiều box nhỏ cho cùng object.

Nếu tăng IoU threshold lên 0.30 như config #2 (0.40/0.30), merge được nhiều
partial box hơn → AP_micro tăng (0.6030), nhưng AP_large vẫn thấp (0.5059)
vì average box không khôi phục được kích thước thật.

#### 3. Overlap between adjacent tiles is real overlap, not duplicate

Hai tile liền kề có 64px overlap. Model trên tile A và tile B dự đoán cùng
object ở overlapping region → 2 box này thực sự là cùng object, nhưng IoU thấp
vì mỗi tile chỉ thấy một phần.

Cần nhận diện: overlap từ adjacent tiles = same object, merge chúng bằng extent
union (không phải average).

### Improvement Plan

3 cải tiến song song:

#### C1. Tile-adjacency aware merging

Trước khi tính IoU để cluster, đánh dấu các box từ **cùng object trên 2 tile
liền kề** bằng spatial proximity trong overlapping zone:

```python
def is_adjacent_tile_overlap(box_a, tile_a, box_b, tile_b, overlap=64):
    """Check if two boxes are from adjacent tiles and overlap in the overlap zone."""
    # Boxes are from different tiles
    if tile_a == tile_b: return False
    # Box centers are in the overlap zone
    # If both predict the same object in overlapping area -> same object
    ...
```

Nếu detect được tile-adjacent overlap → cluster mặc định (không cần IoU threshold).

#### C2. Extent fusion thay vì weighted average

Thay vì `avg_box = weighted_mean(boxes)`, dùng **weighted extent**:

```python
def extent_fusion(boxes, scores):
    """Take min/max extent weighted by score confidence.
    
    x1 = weighted_min(x1_i, w=s_i)  # low-confidence boxes contribute less to extent
    x2 = weighted_max(x2_i, w=s_i)
    """
    weights = scores / scores.sum()
    # x1: weighted by confidence — high-conf boxes have more say on boundary
    x1 = (boxes[:,0] * weights).sum()
    y1 = (boxes[:,1] * weights).sum()
    x2 = (boxes[:,2] * weights).sum()
    y2 = (boxes[:,3] * weights).sum()
    return (x1, y1, x2, y2)
```

Nhưng cái này vẫn có vấn đề — weight average của 2 partial box không khôi phục
được full extent. Cần cách khác.

**Phương án đúng:** Extent hull — lấy min x1, min y1, max x2, max y2 từ tất cả
box trong cluster, nhưng có kiểm tra outlier (1 box lệch xa thì ignore).

```python
def extent_hull(boxes, scores, iqr_mult=1.5):
    """Convex hull of all boxes in cluster, with outlier rejection."""
    if len(boxes) <= 1:
        return boxes.mean(0) if len(boxes) == 1 else None
    
    # IQR-based outlier rejection on each coordinate
    x1s, y1s, x2s, y2s = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    
    def robust_min_max(vals, mult=1.5):
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - mult*iqr, q3 + mult*iqr
        valid = (vals >= lo) & (vals <= hi)
        if valid.any():
            return vals[valid].min(), vals[valid].max()
        return vals.min(), vals.max()
    
    x1, x2 = robust_min_max(x1s)
    y1, y2 = robust_min_max(y1s)
    
    return torch.tensor([x1, y1, x2, y2])
```

#### C3. Scale-adaptive IoU threshold

Object lớn → IoU threshold cao hơn (khó merge hơn), object nhỏ → threshold
thấp hơn (dễ merge hơn — vì tiny object ít khi có duplicate across tiles):

```python
def adaptive_iou_thr(box_area, base_thr=0.50):
    """Scale-adaptive IoU threshold.
    
    Tiny (<256 px^2): thr = 0.25  (easy merge — rare duplicates)
    Small (256-1024): thr = 0.40
    Medium (1024-4096): thr = 0.55
    Large (>4096): thr = 0.65  (hard merge — avoid over-merging)
    """
    if box_area < 256:   return 0.25
    if box_area < 1024:  return 0.40
    if box_area < 4096:  return 0.55
    return 0.65
```

### Implementation Plan

File: `common/wbf.py` — self-contained WBF module

```python
class SmartWBF:
    def __init__(self, base_iou_thr=0.55, overlap_px=64):
        self.base_iou_thr = base_iou_thr
        self.overlap_px = overlap_px
    
    def fuse(self, tile_boxes, tile_coords, img_size):
        """Fuse predictions from multiple tiles into full-image predictions.
        
        Steps:
        1. Remap all boxes to full-image coords
        2. Compute scale-adaptive IoU threshold per box pair
        3. Cluster using adaptive threshold
        4. For each cluster: extent hull fusion
        5. Score fusion: max score in cluster
        """
```

### Hypotheses to Test

| ID | Hypothesis | Test |
|:--:|-----------|------|
| H4.3 | Extent hull recovers large-object AP better than weighted average | Compare extent_hull vs wbf_avg on AP_large |
| H4.4 | Scale-adaptive IoU threshold improves micro-AP without hurting large-AP | Grid search base_thr with adaptive on/off |
| H4.5 | Tile-adjacency pre-clustering reduces reliance on IoU for adjacent tiles | Ablation: with/without adjacency check |

### Grid Search Space

```python
GRID = {
    "base_iou_thr": [0.40, 0.50, 0.55, 0.60],
    "score_thr":    [0.10, 0.20, 0.30],
    "fusion_mode":  ["extent_hull", "weighted_avg"],
    "adaptive_thr": [True, False],
}
# 4 × 3 × 2 × 2 = 48 configs
```

### Expected Outcome

- **extent_hull + adaptive_thr**: AP_large phục hồi về mức standalone (~0.60-0.65)
  trong khi giữ AP_micro được cải thiện (~0.57-0.60)
- **mAP(scale) target**: > 0.62 (vượt cả standalone và cascade hiện tại)
