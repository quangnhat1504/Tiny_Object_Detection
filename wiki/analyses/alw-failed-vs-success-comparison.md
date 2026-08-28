---
title: ALW Failed vs Success Comparison
type: analysis
created: 2026-06-05
updated: 2026-06-05
sources: [working/code/4_alw.ipynb, working/code/tod-alw.ipynb]
tags: [alw, analysis, comparison]
---

# So Sánh: Tại Sao ALW Ban Đầu Thất Bại (4_alw.ipynb) vs Thành Công (tod-alw.ipynb)

## Kết Quả Đối Lập

| Implementation | mAP(scale) | COCO AP@50:75 | Det/img | Status |
|----------------|------------|---------------|---------|--------|
| **4_alw.ipynb** (bạn code) | **0.1822** | 0.0652 | **599.20** | ❌ THẤT BẠI |
| **tod-alw.ipynb** (clean) | **0.5439** | 0.1893 | ~909* | ✅ THÀNH CÔNG |

*Note: Det/img cao trong tod-alw do score_thresh=0.30 thấp hơn (vs 0.40), nhưng mAP vẫn cao → quality tốt

**Performance gap: 3× improvement!** (0.5439 vs 0.1822)

---

## Nguyên Nhân Thất Bại: Over-Engineering

### ❌ VẤN ĐỀ 1: Metric Formula Quá Phức Tạp

**4_alw.ipynb (FAILED)** - "RG-Robust ALW":
```python
# Lines 169-203: Reliability-Gated + Charbonnier robust
def _core_dist(cxa, cya, wa, ha, cxb, cyb, wb, hb):
    # Position: OK (giống tod-alw)
    dx = cxa - cxb; dy = cya - cyb
    Sx = (wa * wa + wb * wb) * 0.5
    Sy = (ha * ha + hb * hb) * 0.5
    pos = dx*dx/Sx + dy*dy/Sy
    
    # ⚠️ PROBLEM: Shape term quá phức tạp!
    gt_size = sqrt(wb * hb)
    size_gate = (gt_size / 16.0).clamp(0, 1)  # reliability gate
    
    # Variable shape lambda theo size
    shape_lambda = 0.35 + 0.65 * (size_gate ** 1.5)
    
    # Charbonnier robust eps theo size
    robust_eps = 0.001 + 0.35 * (1.0 - size_gate)
    
    # Charbonnier: sqrt(x² + eps²) - eps
    lr_w = log(wa/wb).abs()
    lr_h = log(ha/hb).abs()
    shp_w = sqrt(lr_w² + robust_eps²) - robust_eps
    shp_h = sqrt(lr_h² + robust_eps²) - robust_eps
    shp = shape_lambda * (shp_w + shp_h)
    
    return sqrt(pos + shp)
```

**Vấn đề:**
1. **5 hyperparameters mới**: `RELIABILITY_THR=16.0`, `LAMBDA_MIN=0.35`, `LAMBDA_POWER=1.5`, `EPS_MIN=0.001`, `EPS_MAX=0.35`
2. **Size-dependent behavior** → không consistent
3. **Charbonnier smoothing** → giảm gradient cho small objects (ngược với mục đích!)
4. **Variable shape weight** → micro objects bị under-weight shape term

**tod-alw.ipynb (SUCCESS)** - Pure ALW:
```python
# Lines 306-360: Simple and correct
def alw_similarity(xn, yn, wn, hn, xg, yg, wg, hg, beta=8.0):
    eps = 1e-6
    wn = wn.clamp(min=eps); hn = hn.clamp(min=eps)
    wg = wg.clamp(min=eps); hg = hg.clamp(min=eps)
    
    # Position terms - anisotropic
    dx = xn.unsqueeze(1) - xg.unsqueeze(0)
    dy = yn.unsqueeze(1) - yg.unsqueeze(0)
    Sx = (wn.unsqueeze(1)**2 + wg.unsqueeze(0)**2) / 2.0
    Sy = (hn.unsqueeze(1)**2 + hg.unsqueeze(0)**2) / 2.0
    pos_x = dx**2 / Sx.clamp(min=eps)
    pos_y = dy**2 / Sy.clamp(min=eps)
    
    # Shape terms - clean log-ratio
    shape_w = torch.log(wn.unsqueeze(1) / wg.unsqueeze(0))**2
    shape_h = torch.log(hn.unsqueeze(1) / hg.unsqueeze(0))**2
    
    # ALW distance
    alw_sq = (pos_x + pos_y + shape_w + shape_h).clamp(min=0.0)
    alw = alw_sq.sqrt().clamp(max=30.0/beta)
    return torch.exp(-beta * alw)
```

**Ưu điểm:**
- Chỉ 1 hyperparameter: `beta=8.0`
- Consistent behavior across scales
- Full gradient cho tất cả objects
- Mathematically pure

---

### ❌ VẤN ĐỀ 2: Dynamic Top-K Quá Nhiều Positives

**4_alw.ipynb (FAILED)**:
```python
# Lines 383-438: Dynamic k với quality gate
def get_dynamic_k(gt_boxes):
    sz = sqrt(w * h)
    # ⚠️ PROBLEM: Quá nhiều positives cho small objects!
    if sz < 6:  return 6   # micro
    if sz < 16: return 5   # tiny
    if sz < 64: return 4   # small
    return 3               # large (= RFLA_K baseline)

def hierarchical_label_assignment(...):
    dyn_k = get_dynamic_k(gt_boxes)
    
    for gi in range(M):
        k_gi = int(dyn_k[gi])  # 3-6 anchors tùy scale
        top_scores, top_idx = sim[:, gi].topk(k_gi)
        
        # Quality gate: chỉ giữ anchor có sim >= 0.60 * best
        keep = (top_scores >= top_scores[0] * 0.60)
        matched_gt[top_idx[keep]] = gi
```

**Tại sao thất bại:**
1. **Micro objects (n=1009) có k=6** → 6072 positives → quá nhiều false positives
2. **Quality gate 0.60 quá lỏng** → accept cả anchors kém
3. **599 det/img** → massive over-detection
4. **mAP=0.1822** → precision collapse

**tod-alw.ipynb (SUCCESS)**:
```python
# Lines 363-403: Fixed k=3, stricter assignment
def hierarchical_label_assignment(
    rfd_scores, xn, yn, wn, hn, xg, yg, wg, hg,
    k=3, beta=0.9  # ⭐ FIXED k=3 cho mọi scale
):
    # Pass 1: Top-3 anchors cho mỗi GT
    for gi in range(M):
        _, top_idx = rfd_scores[:, gi].topk(min(k, N))  # k=3
        matched_gt[top_idx] = gi
    
    # Pass 2: Expand anchors (scale beta=0.9), add thêm nếu cần
    rfd2 = compute_rfd(xn, yn, wn*beta, hn*beta, xg, yg, wg, hg)
    for gi in range(M):
        if (~assigned_mask).sum() == 0: break
        sc = rfd2[:, gi].clone()
        sc[assigned_mask] = -1.0
        best = sc.argmax()
        # ⭐ STRICTER: chỉ add nếu chưa đủ k=3 anchors
        if sc[best] > 0 and (r1 == gi).sum() < k:
            matched_gt[best] = gi
```

**Tại sao thành công:**
1. **k=3 cho mọi scale** → consistent, không over-assign
2. **Pass 2 chỉ add khi chưa đủ k** → controlled growth
3. **Simpler logic** → ít bugs, dễ debug
4. **mAP=0.5439** → precision/recall cân bằng

---

### ❌ VẤN ĐỀ 3: Vẫn Dùng Similarity-Based Loss

**4_alw.ipynb (FAILED)**:
```python
# Line 514: Loss = 1 - similarity
sim = metric_sim_pair(pred_boxes, tgt_boxes)
loss_box = (1.0 - sim).mean()  # ⚠️ exp(-β*d) trong gradient!
```

**Gradient problem:**
```python
# Với β=8, distance=0.5:
sim = exp(-8 * 0.5) = 0.018
loss = 1 - 0.018 = 0.982
gradient = ∂loss/∂d = β * exp(-β*d) = 8 * 0.018 = 0.144

# → Gradient rất nhỏ cho large errors!
```

**tod-alw.ipynb (SUCCESS)**:
```python
# Line 557: Direct distance loss
def alw_bbox_loss(pred_boxes, target_boxes):
    return _alw_pair_distance(pred_boxes, target_boxes)
    # → raw ALW distance, không qua exp()

# Gradient = ∂ALW/∂d = 1.0 (luôn mạnh)
```

---

## Nguyên Tắc Thiết Kế

### ❌ 4_alw.ipynb: "Clever" Over-Engineering

```
Ý tưởng: "Micro objects nhỏ quá → cần special handling"
→ Reliability gate
→ Charbonnier smoothing  
→ Dynamic k=6 cho micro
→ Quality gate 0.60

Kết quả: Complexity explosion, unstable, failed
```

### ✅ tod-alw.ipynb: KISS Principle

```
Ý tưởng: "Implement ALW đúng toán học, let it work"
→ Pure log-ratio (scale-invariant)
→ Anisotropic normalization (geometry-correct)
→ Direct distance loss (strong gradients)
→ Fixed k=3 (simple, proven)

Kết quả: Clean, stable, mAP=0.5439
```

---

## Bài Học Quan Trọng

### 1. **Đừng "Fix" Cái Không Hỏng**

ALW đã mathematically correct. Thêm "improvements" như:
- Reliability gate
- Charbonnier smoothing
- Dynamic k

→ Không improve mà làm hỏng!

### 2. **Simple > Clever**

```python
# BAD: 5 hyperparameters
shape_lambda = ALW_SHAPE_LAMBDA_MIN + \
    (1.0 - ALW_SHAPE_LAMBDA_MIN) * (size_gate ** ALW_SHAPE_LAMBDA_POWER)

# GOOD: 0 extra hyperparameters
shape_w = torch.log(wn / wg)**2
```

### 3. **Trust the Math**

ALW formula đã scale-invariant thông qua log-ratio. Không cần thêm size-dependent gating.

### 4. **Less is More**

- 4_alw.ipynb: 10+ hyperparameters → mAP=0.1822
- tod-alw.ipynb: 1 hyperparameter (β=8) → mAP=0.5439

### 5. **Debug Early, Debug Often**

Nếu mAP=0.1822 và det/img=599, đó là signal rõ ràng:
- **Not underfitting** → model đang detect nhiều
- **Precision collapse** → false positives quá nhiều
- **Root cause**: Assignment quá lỏng (k=6, quality=0.60)

---

## Checklist Thiết Kế Metric

✅ **DO:**
- Implement công thức gốc đúng toán học
- Keep it simple
- Use proven techniques (k=3 RFLA)
- Test baseline trước khi thêm "improvements"

❌ **DON'T:**
- Add complexity "just in case"
- Over-engineer cho edge cases chưa test
- Trust intuition hơn math
- Skip ablation studies

---

## Kết Luận

**4_alw.ipynb thất bại không phải do ALW metric xấu**, mà do:
1. Over-complicated formula (reliability gate + Charbonnier)
2. Too many positives (k=6 cho micro, quality=0.60)
3. Similarity-based loss (weak gradients)

**tod-alw.ipynb thành công vì:**
1. Pure ALW formula (mathematically correct)
2. Simple assignment (k=3, proven)
3. Direct distance loss (strong gradients)

**Performance: 0.5439 vs 0.1822 = 3× improvement**

Bài học: **"Premature optimization is the root of all evil"** - Donald Knuth. Đừng cố "improve" một cái đã đúng. Keep it simple, stupid!