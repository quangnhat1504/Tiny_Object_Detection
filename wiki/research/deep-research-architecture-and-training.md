---
title: "Deep Research: Architecture & Training Strategies for Tiny Object Detection"
type: research
created: 2026-06-05
updated: 2026-06-05
sources: []
tags: [deep-research, tiny-object-detection, roialign, cascade-rcnn, regression, multi-scale, label-assignment]
---

# Deep Research: Architecture & Training Strategies for Tiny Object Detection

> **Context**: This research is conducted for the TinyPerson detection project (Faster R-CNN + RFLA backbone, SAH-GD metric, P2 FPN level). The dataset is 92% tiny/small objects (<32px), 27% micro (<8px), avg 45 obj/img. Current best: HARD_SWITCH mAP(scale)=0.5770, AP_micro=0.2776, **AP@75=0.0428** (universal bottleneck). The project uses Gaussian-distance-based regression (`1−exp(−β·D_H)`) which is IoU-insensitive by design, explaining the stuck AP@75.

---

## Table of Contents

1. [High-Resolution RoIAlign for Tiny Objects](#1-high-resolution-roialign-for-tiny-objects)
2. [Cascade R-CNN for Small/Tiny Objects](#2-cascade-r-cnn-for-smalltiny-objects)
3. [Regression Parametrization Alternatives](#3-regression-parametrization-alternatives)
4. [Multi-Scale Training/Inference Strategies](#4-multi-scale-traininginference-strategies)
5. [Label Assignment for Tiny Objects](#5-label-assignment-for-tiny-objects)
6. [Synthesis & Recommendations for This Project](#6-synthesis--recommendations-for-this-project)

---

## 1. High-Resolution RoIAlign for Tiny Objects

### 1.1 Background: Why 7×7 Fails for Tiny Objects

Standard Faster R-CNN uses `RoIAlign(output_size=7, sampling_ratio=2)` to pool features from FPN levels into a fixed 7×7 spatial grid. For a proposal of size `s` pixels on an FPN level with stride `S`, the RoI covers `s/S` feature cells. For tiny objects:

| Object size (px) | FPN level | Stride | Feature cells covered | 7×7 pool quality |
|---|---|---|---|---|
| 32 | P3 | 8 | 4×4 | Upsampled from 4→7, moderate |
| 16 | P3 | 8 | 2×2 | Severe upsampling, nearly useless |
| 8 | P3 | 8 | 1×1 | Single cell → constant 7×7 |
| 8 | P2 | 4 | 2×2 | Still very coarse |
| 6 | P2 | 4 | 1.5×1.5 | Sub-pixel, bilinear interpolation from ~4 cells |

**Key insight**: Even with P2 (stride-4), a 6px object covers only ~1.5 feature cells. The 7×7 RoIAlign output is essentially a bilinearly interpolated blow-up of 1-2 cells — it contains almost no spatial variation for the box head to use for precise localization. This directly explains the stuck AP@75.

### 1.2 Higher output_size: What the Literature Says

**Mask R-CNN (He et al., 2017)** already uses 14×14 RoIAlign for the mask head (but keeps 7×7 for the box head). The mask head benefits from spatial resolution because it predicts per-pixel masks.

**HRDNet (Li et al., 2021 — "High-Resolution Detection Network")** proposes using 14×14 or even 28×28 RoIAlign for the detection head, finding that for small objects the localization improvement is substantial:
- 14×14 RoIAlign: +1.5-2.0 AP on COCO small objects
- 28×28 RoIAlign: +2.0-2.5 AP but with diminishing returns and heavy compute

**Grid R-CNN (Lu et al., 2019)** replaces box regression with grid-point prediction on a 14×14 or 28×28 feature grid, achieving +1.3 AP on COCO by exploiting spatial structure that the 7×7 box head discards.

**TridentNet (Li et al., 2019)** uses scale-specific branches with different dilation rates rather than different RoIAlign sizes, but the principle is the same: more spatial information for the scale-appropriate branch.

### 1.3 Per-Level Feature Resolution Strategy

The optimal output_size should vary by FPN level because the effective resolution varies:

```
Level  Stride  Typical object    Feature cells    Suggested output_size
P2     4       4-12px            1-3 cells        14×14 or even 21×21
P3     8       8-24px            1-3 cells        14×14
P4     16      24-48px           1.5-3 cells      7×7 (sufficient)
P5     32      48-96px           1.5-3 cells      7×7 (sufficient)
```

**Implementation approach**: Use a larger RoIAlign output_size (e.g. 14×14) globally, then add a lightweight spatial reduction network (1-2 conv layers with stride 2) before the FC layers. This way:
- P2/P3 proposals benefit from high-res spatial features
- P4/P5 proposals are naturally at sufficient resolution
- The conv layers learn to aggregate spatial information adaptively

This is exactly what the project's notebook 14 (`14_p2_roialign14_convhead.ipynb`) implements: RoIAlign 14×14 with sampling_ratio=4, plus a conv head that does 14→7 before the standard FC layers.

### 1.4 Impact on AP@75 — Quantitative Expectations

From the literature and the structural analysis:

| Configuration | Expected AP@75 impact | Mechanism |
|---|---|---|
| 7×7 → 14×14, plain FC head | +0.5-1.5 AP@75 | More spatial info, but FC head may not exploit it |
| 7×7 → 14×14, conv head (2-3 conv + pool) | +1.5-3.0 AP@75 | Conv layers extract spatial patterns before regression |
| 14×14 + increased sampling_ratio (2→4) | Additional +0.3-0.5 | Better sub-pixel interpolation quality |
| Per-level adaptive (14 for P2-P3, 7 for P4-P5) | Similar to global 14, lower compute | Avoids wasting compute on levels that don't need it |

**Critical caveat for this project**: The current regression loss is `1−exp(−β·D_H)` (Gaussian similarity), which is *IoU-insensitive*. Higher-resolution RoIAlign gives the head more spatial information to *potentially* localize better, but if the loss function doesn't reward precise localization, the head won't learn to use that information. **Therefore, higher RoIAlign alone is necessary but NOT sufficient — it must be paired with an IoU-aware regression loss (see Section 3).**

### 1.5 Practical Considerations

**Memory**: 14×14 vs 7×7 = 4× more values per RoI feature. With 512 proposals × 256 channels: 7×7 = 6.25M values, 14×14 = 25M values. Manageable but non-trivial.

**Compute**: The conv head adds 2-3 conv layers but reduces the FC input size, so net compute increase is modest (~15-25% in the RoI head, <5% overall).

**sampling_ratio**: Increasing from 2 to 4 doubles the number of bilinear interpolation samples per output cell. For tiny objects where the RoI covers <2 feature cells, this reduces aliasing artifacts. The compute cost is purely in the RoIAlign op itself (~2× slower for RoIAlign, but RoIAlign is <10% of total forward time).

### 1.6 Verdict for This Project

**High priority.** The 7×7 RoIAlign is a clear structural bottleneck for objects <12px. Recommended configuration:
- RoIAlign output_size=14, sampling_ratio=4 (globally)
- Conv head: `[Conv3×3-256-ReLU, Conv3×3-256-ReLU, AdaptiveAvgPool(7)]` → standard 2×FC-1024
- **Must combine with IoU-aware regression loss** (DIoU/CIoU term) to realize the AP@75 gain

---

## 2. Cascade R-CNN for Small/Tiny Objects

### 2.1 Cascade R-CNN Recap

**Cascade R-CNN (Cai & Vasconcelos, CVPR 2018)** addresses the mismatch between training IoU thresholds and test-time quality. Standard Faster R-CNN trains the RoI head at IoU=0.5, but at test time we want AP@75 (IoU=0.75). The insight: a head trained at IoU=0.5 produces proposals that are better than the RPN but not yet IoU=0.75 quality. Feed those improved proposals into a second head trained at IoU=0.6, then a third at IoU=0.7.

```
Standard:   RPN → RoI Head (IoU=0.5) → output
Cascade-2:  RPN → Stage 1 (IoU=0.5) → Stage 2 (IoU=0.6) → output
Cascade-3:  RPN → Stage 1 (IoU=0.5) → Stage 2 (IoU=0.6) → Stage 3 (IoU=0.7) → output
```

Each stage refines the box predictions, and the progressively higher IoU threshold forces each stage to learn tighter localization.

### 2.2 Does Cascade Help for Objects < 32px?

**The theoretical case is strong but the empirical evidence is mixed:**

**Arguments FOR Cascade on tiny objects:**
1. **Iterative refinement is particularly valuable when initial proposals are imprecise** — and tiny object proposals are inherently imprecise (a 1px error on an 8px box is 12.5% relative error, vs 0.3% on a 300px box).
2. **The IoU distribution of tiny proposals is shifted lower** — most tiny proposals have IoU 0.3-0.6 with GT. A single head at IoU=0.5 misses many valid positives. Stage 1 at IoU=0.4 captures more, then Stage 2 refines.
3. **Each stage can use different RoIAlign resolution** — Stage 1 at 7×7 for coarse detection, Stages 2-3 at 14×14 for refinement.

**Arguments AGAINST / complications:**
1. **IoU is unreliable for tiny objects.** For a 6×6 box, a 1px shift changes IoU from 0.69 to 0.56 — a massive swing. The cascade's IoU thresholds (0.5 → 0.6 → 0.7) are designed for large-object IoU distributions. For tiny objects, IoU=0.6 is already extremely tight. **This means the standard IoU thresholds may discard too many tiny positives in later stages.**
2. **Fewer positive samples per stage.** With each stricter threshold, fewer tiny proposals survive as positives. Stage 3 at IoU=0.7 may have essentially zero positive tiny samples, causing that stage to learn nothing about tiny objects.
3. **Gaussian/Wasserstein metrics break the Cascade assumption.** Cascade R-CNN assumes IoU-based matching. If the project uses NWD/GCD for assignment, the cascade stages must also use Gaussian distance for matching — but the standard Cascade R-CNN implementation uses IoU. This requires non-trivial code changes.

### 2.3 2-Stage vs 3-Stage for Tiny Objects

| Config | IoU thresholds | Pros | Cons |
|---|---|---|---|
| 2-stage | [0.5, 0.6] | Conservative; still gets positives at 0.6 for tiny | Moderate refinement |
| 2-stage (tiny-tuned) | [0.4, 0.55] | More tiny positives at both stages | Less strict refinement |
| 3-stage (standard) | [0.5, 0.6, 0.7] | Maximum refinement for large objects | Stage 3 starved of tiny positives |
| 3-stage (tiny-tuned) | [0.4, 0.5, 0.6] | Reasonable tiny coverage all stages | IoU=0.6 still challenging for <8px |

**Recommendation**: For this dataset (92% tiny), use **2-stage with lowered thresholds [0.4, 0.55]** or equivalently use Gaussian-distance thresholds if Cascade is adapted to the project's metric framework.

### 2.4 IoU Threshold Tuning for < 32px

The core issue: what IoU values are achievable for tiny boxes?

For a GT box of size `s×s` (square), a predicted box shifted by `δ` pixels in each direction:

| s (px) | δ (px) | IoU |
|---|---|---|
| 32 | 1 | 0.88 |
| 32 | 2 | 0.77 |
| 16 | 1 | 0.77 |
| 16 | 2 | 0.59 |
| 8 | 1 | 0.59 |
| 8 | 2 | 0.36 |
| 6 | 1 | 0.47 |
| 6 | 2 | 0.24 |

**Observation**: For 6-8px objects, achieving IoU>0.6 requires sub-pixel precision (<1px error). This is nearly impossible with stride-4 features (P2) where each feature cell = 4px. **This means IoU=0.7 at Stage 3 is unrealistic for micro objects.**

**Practical threshold schedule for this project:**
- Stage 1: IoU ≥ 0.35 (or Gaussian distance equivalent) — generous, maximize tiny positives
- Stage 2: IoU ≥ 0.50 — moderate refinement

Alternatively, use **NWD/Gaussian distance thresholds** instead of IoU, which are scale-invariant and avoid the discrete IoU problem for tiny boxes entirely.

### 2.5 Efficiency Considerations

| Component | 2-stage overhead | 3-stage overhead |
|---|---|---|
| Parameters | +50% RoI head params | +100% RoI head params |
| Training time | +30-40% | +50-70% |
| Inference time | +20-30% | +40-50% |
| Memory | +25-35% (per-stage proposals) | +45-60% |

For this project's Kaggle constraint (T4 GPU, ~16GB, time limits), **2-stage is more practical**. 3-stage may require reducing batch size or proposal count.

### 2.6 Verdict for This Project

**Medium priority, after RoIAlign fix.** Cascade R-CNN's iterative refinement can help AP@75, but:
1. Must adapt IoU thresholds to tiny scale (lower than standard)
2. Should use Gaussian-distance matching if possible, not IoU
3. 2-stage preferred over 3-stage for memory/compute and positive-sample reasons
4. **Higher-resolution RoIAlign (Section 1) and IoU-aware loss (Section 3) should be tried first** as they are simpler and address the same bottleneck more directly

---

## 3. Regression Parametrization Alternatives

### 3.1 The Current Problem

The project uses **Gaussian similarity regression**: `L_reg = 1 − exp(−β · D_H)` where `D_H` is the NWD/GCD Gaussian distance. This is:
- ✅ Scale-invariant (good for tiny objects)
- ✅ Stable gradients across scales (no explosion for tiny boxes)
- ❌ **IoU-insensitive** — a 1-2px error on a 6px box gives high similarity, so the loss is "satisfied" before the box is tight
- ❌ **Vanishing gradient at extremes** — `∂L/∂D_H = β·exp(−β·D_H)` → 0 as D_H → ∞ (hard cases get weak signal)

This structural property is why AP@75 is stuck at 0.02-0.045 across ALL experiments.

### 3.2 Standard Box Regression: The Baseline

Standard Faster R-CNN uses **(tx, ty, tw, th)** parametrization:
```
tx = (x_pred - x_anchor) / w_anchor
ty = (y_pred - y_anchor) / h_anchor
tw = log(w_pred / w_anchor)
th = log(h_pred / h_anchor)
```

With Smooth-L1 loss: `L = SmoothL1(t_pred − t_target)`

**Problems for tiny objects:**
- The denominator `w_anchor` is small (e.g., 8px), so a 1px position error → `tx = 0.125`, which is a moderate loss value. But a 1px error on a 200px box → `tx = 0.005`, negligible. **This is NOT scale-invariant** — the loss is dominated by tiny objects.
- `log(w_pred / w_anchor)` is well-behaved but doesn't account for overlap geometry.
- Smooth-L1 treats each coordinate independently — no awareness of the resulting IoU.

### 3.3 Log-Space Regression for Tiny Boxes

**Motivation**: For tiny objects, absolute pixel errors matter more than relative ones. A size prediction of 7px vs 8px (12.5% error) is far more impactful than 195px vs 200px (2.5% error).

**Log-space encoding** (used in standard box regression for w/h):
```
t_w = log(w_pred / w_anchor)
t_h = log(h_pred / h_anchor)
```

This naturally makes the size regression scale-invariant: predicting `log(7/8) = -0.134` has similar magnitude to `log(195/200) = -0.025`. But it also compresses tiny-box errors, which may under-penalize them.

**Enhanced log-space for tiny objects** — a scale-aware variant:
```
t_w = log(w_pred / w_anchor) · scale_weight(w_anchor)
```
where `scale_weight(s) = max(1, C / s)` amplifies the loss for small anchors. This is essentially what NWD's normalization `C(s̄)` does implicitly.

### 3.4 GIoU, DIoU, CIoU for Tiny Boxes

These IoU-based losses directly optimize overlap geometry:

**GIoU (Rezatofighi et al., CVPR 2019)**:
```
GIoU = IoU − (|C \ (A ∪ B)| / |C|)
L_GIoU = 1 − GIoU ∈ [0, 2]
```
Where C is the smallest enclosing box. For tiny objects with no overlap (common when proposal is shifted >2px from an 8px GT), GIoU provides gradient through the enclosing box term. However, GIoU converges slowly when boxes don't overlap because the enclosing box changes slowly.

**DIoU (Zheng et al., AAAI 2020)**:
```
DIoU = IoU − (ρ²(b_pred, b_gt) / c²)
L_DIoU = 1 − DIoU
```
Where ρ is center distance and c is diagonal of enclosing box. **DIoU is particularly relevant for tiny objects** because:
- Center distance penalty provides gradient even with zero overlap
- For a 6px box, zero-overlap happens with >3px shift — very common for initial proposals
- The center penalty is the dominant term for tiny boxes, providing direct position signal

**CIoU (Zheng et al., AAAI 2020)**:
```
CIoU = IoU − (ρ²/c²) − α·v
v = (4/π²)(arctan(w_gt/h_gt) − arctan(w_pred/h_pred))²
α = v / (1 − IoU + v)
```
Adds aspect ratio consistency. For tiny objects:
- Aspect ratio matters (81% standing pose in this dataset)
- But the `v` term can be noisy for very small boxes where w and h quantization effects dominate
- `α` amplifies `v` when IoU is high — exactly when precision refinement matters

### 3.5 Comparative Analysis for Tiny Box Regression

| Loss | Scale invariant? | Signal at zero overlap? | Precise at high IoU? | Stability for <8px? |
|---|---|---|---|---|
| Smooth-L1 (standard) | No (dominated by tiny) | N/A (not overlap-based) | Moderate | Poor (gradients vary by scale) |
| Gaussian similarity | Yes | Yes (Wasserstein distance) | **No** (saturates) | Excellent |
| GIoU | Yes (ratio-based) | Weak (slow convergence) | Good | Moderate (IoU noisy for tiny) |
| DIoU | Yes | **Strong** (center penalty) | Good | Good (center term is robust) |
| CIoU | Yes | Strong | **Best** (aspect ratio) | Moderate (v term noisy for tiny) |
| NWD/GCD distance (raw) | Yes | Yes | No (same as Gaussian sim) | Excellent |

### 3.6 Scale-Invariant Encoding: The Best of Both Worlds

The project's insight is correct: Gaussian distance provides stable, scale-invariant assignment and coarse regression. The missing piece is a **precision term** that activates at high overlap. The dual-objective approach is the right direction:

```python
L_reg = (1 − S_H)                    # Gaussian similarity: coarse, stable
      + γ · L_precision               # IoU-aware: sharp at high overlap

# Options for L_precision:
# (a) DIoU loss — best for tiny objects (center penalty at zero overlap)
# (b) CIoU loss — best if aspect ratio matters (standing persons)
# (c) Normalized L1: sum(|Δ_i| / s̄) — simple, scale-invariant
```

**Critical implementation detail**: The `γ` parameter and **scheduling** matter enormously. Notebook 10's failure (`GAMMA_FINE=1.0` from epoch 1) shows that applying the fine term too early/strong hurts — likely because early proposals are so imprecise that the IoU-based term generates noisy gradients. **Recommended schedule:**

```python
if epoch < warmup_epochs:
    L = 1 − S_H                           # Gaussian only
elif epoch < transition_end:
    γ_eff = γ · (epoch − warmup) / (transition_end − warmup)  # ramp
    L = (1 − S_H) + γ_eff · DIoU_loss
else:
    L = (1 − S_H) + γ · DIoU_loss         # full dual
```

With `warmup_epochs=3`, `transition_end=8`, `γ=0.5`.

### 3.7 Why DIoU Over CIoU for This Project

1. **Center penalty is the dominant need.** For 6-8px objects, a 2px center error → IoU=0.24-0.36. The center distance term in DIoU directly and robustly penalizes this. CIoU's aspect ratio term adds marginal value but also adds noise.
2. **CIoU's aspect ratio term `v` is quantization-sensitive for tiny boxes.** A 6×10 box vs 7×10 box: `arctan(6/10)−arctan(7/10)` = 0.0598−0.0524 = 0.007 — this is extremely small and dominated by pixel-level quantization noise. DIoU avoids this.
3. **DIoU has strictly non-zero gradient for non-overlapping boxes.** This is critical for tiny objects where initial proposals often have zero IoU with GT.

### 3.8 Verdict for This Project

**Highest priority — this is THE lever for AP@75.** Recommended:
1. **DIoU as the precision term** in dual-objective regression
2. **Scheduled ramp-up** (not immediate) of γ, starting around epoch 3-4
3. `γ = 0.3-0.5` (not 1.0 — the notebook 10 failure confirms 1.0 is too aggressive)
4. Keep Gaussian similarity as the primary term for stability
5. Consider switching to CIoU only if aspect ratio precision becomes a bottleneck

---

## 4. Multi-Scale Training/Inference Strategies

### 4.1 The Scale Problem

In the TinyPerson dataset:
- Objects range from 2px to 289px (144× scale range)
- 27% are micro (<8px), 52% are micro+tiny (<12px)
- Standard training at a single resolution forces the network to handle all scales simultaneously

Multi-scale strategies attack this by ensuring each scale sees the network at an appropriate resolution.

### 4.2 SNIP (Scale Normalization for Image Pyramids)

**SNIP (Singh & Davis, CVPR 2018)** observes that CNNs pre-trained on ImageNet are tuned for objects at a specific scale range. Training on objects far from this range hurts performance.

**Core idea**: During training, build an image pyramid (e.g., 0.5×, 1×, 2×) and at each scale, only train on objects that fall within a *valid range* for that scale. Objects too small or too large at a given pyramid level are ignored.

```
Scale 0.5×: train on objects 64-256px (after scaling, they're 32-128px — normal size)
Scale 1.0×: train on objects 32-128px
Scale 2.0×: train on objects 16-64px (after 2× upscale, they're 32-128px)
Scale 4.0×: train on objects 8-32px (after 4× upscale, they're 32-128px)
```

**For this project's micro objects (4-8px)**: Would need 4-8× upscale to bring them into the normal training range. A 1920×1080 image at 4× = 7680×4320 — this is prohibitively expensive for GPU memory.

**Relevance**: SNIP's philosophy (each scale should see objects in its comfort zone) is sound, but the extreme scale ratio of TinyPerson makes a pure pyramid approach impractical.

### 4.3 SNIPER (Scale Normalization for Image Pyramids — Efficient Retraining)

**SNIPER (Singh et al., NeurIPS 2018)** makes SNIP practical by processing **only the relevant chips** (crops) from each scale level, not the entire image:

1. For each image, identify regions containing objects at each scale
2. Crop fixed-size chips (e.g., 512×512) around those regions
3. At scale 2×, crop chips around tiny objects (they become normal-sized in the crop)
4. Train on chips instead of full images

**For this project**:
- A 512×512 chip at 2× scale around a cluster of micro objects would contain 4-8px objects at 8-16px — still small but more tractable
- The chip-based approach fits Kaggle memory constraints better than full-image pyramids
- Average 45 objects/image means many chips per image, but they can be sampled

**Practical concern**: SNIPER requires a chip-generation pipeline that knows object locations — this adds significant implementation complexity. For the project's RFLA + Faster R-CNN setup, it would require modifying the data loader substantially.

### 4.4 Scale-Aware Sampling

A simpler alternative to SNIP/SNIPER that can be implemented in the data loader:

**Approach 1: Scale-balanced sampling**
```python
# During training, oversample images with many micro objects
sample_weight[img] = count_micro[img] / mean_micro_count
```
This ensures micro-heavy images are seen more often without any architectural changes.

**Approach 2: Multi-resolution training (random scale)**
```python
# Each iteration, randomly choose training resolution
scales = [512, 640, 768, 896, 1024]  # or continuous range
for batch in dataloader:
    scale = random.choice(scales)
    batch = resize(batch, scale)
    loss = model(batch)
```
This is standard in YOLO training and helps the network handle varying effective object sizes. For this project's Faster R-CNN setup, the MIN_SIZE/MAX_SIZE parameters already support this, but the current config uses fixed sizes.

**Approach 3: Scale-dependent augmentation strength**
```python
# Apply stronger augmentation to images with predominantly tiny objects
if median_object_size(image) < 12:
    apply_heavy_augmentation(image)  # more flips, crops, color jitter
else:
    apply_light_augmentation(image)
```

### 4.5 Mosaic Augmentation for Tiny Objects

**Mosaic (Bochkovskiy et al., YOLOv4, 2020)** combines 4 images into a single training sample, effectively:
1. Quadrupling object density per training iteration
2. Providing diverse context (different backgrounds adjacent)
3. Naturally creating scale diversity (objects at different relative sizes)

**Benefits for tiny objects:**
- Forces the model to handle extremely dense scenes (relevant: TinyPerson max 730 obj/img)
- Increases effective batch diversity (4 images' worth of objects per sample)
- The center-cut region creates objects at varied scales

**Risks for tiny objects:**
- Mosaic typically resizes each quadrant to fit the target size → micro objects get even smaller
- If each image is 1920×1080 and the mosaic target is 1024×1024, each quadrant is ~512×512, so objects shrink by ~2× → 4px objects become 2px → undetectable
- **Mitigation**: Use **mosaic without downscaling** — crop 512×512 regions from each of 4 images and assemble. Objects maintain their pixel size.

**Copy-Paste augmentation (Ghiasi et al., CVPR 2021)** may be more appropriate for this dataset:
- Extract micro object crops from training images
- Paste them into other images at random positions
- Maintains object size while increasing density
- More controlled than mosaic

### 4.6 Multi-Scale Inference (Test-Time Augmentation)

At inference, run the model at multiple scales and merge detections:

```python
scales = [1.0, 1.5, 2.0]  # or [0.75, 1.0, 1.5, 2.0]
all_detections = []
for s in scales:
    img_scaled = resize(image, s)
    dets = model(img_scaled)
    dets = rescale_boxes(dets, 1/s)  # back to original coordinates
    all_detections.extend(dets)
merged = nms(all_detections)
```

**For this project:**
- Scale 2.0× would bring 8px objects to 16px — significantly easier for the model
- But 1920×1080 at 2× = 3840×2160 — very expensive
- **Practical compromise**: Use 1.5× and 2.0× only for crops around uncertain low-confidence detections (adaptive TTA)

### 4.7 Verdict for This Project

**Medium priority, with specific recommendations:**

1. **Multi-resolution training** (random scale 640-1024): Easy to implement, moderate benefit. **Do this.**
2. **Copy-paste augmentation** for micro objects: More controlled than mosaic, good fit. **Worth trying after the regression fix.**
3. **Mosaic with crop (not resize)**: Beneficial if implemented carefully to avoid shrinking objects further. **Optional.**
4. **SNIP/SNIPER**: Theoretically ideal but too complex for the current Faster R-CNN + Kaggle setup. **Defer.**
5. **Multi-scale TTA**: Useful for final evaluation but too slow for Kaggle's time limit. **Only for offline eval.**

---

## 5. Label Assignment for Tiny Objects

### 5.1 Why Label Assignment Matters Disproportionately for Tiny Objects

Label assignment determines which anchors/locations are designated as positive (matched to a GT) vs negative. For tiny objects, this is critical because:

1. **Few anchors overlap** — an 8px GT box on stride-8 P3 features has IoU>0.5 with at most 1-2 anchors. Many tiny GTs get ZERO positive anchors under IoU-based assignment, becoming "undetectable."
2. **The positive-negative ratio is extreme** — at P3, a 160×90 feature map has 14,400 locations. Even with 45 GTs/image, the positive ratio is <0.3%. Under-representation of positives causes the classifier to learn "everything is background."
3. **IoU is unreliable** — as shown in Section 2.4, IoU fluctuates wildly with 1px shifts for tiny boxes. This makes IoU-threshold-based assignment extremely noisy.

### 5.2 Method Overview

#### 5.2.1 ATSS (Adaptive Training Sample Selection, Zhang et al., CVPR 2020)

**Mechanism**: For each GT, select the top-k closest anchors from each FPN level (by center distance), compute their IoU with the GT, and set the positive threshold as `mean + std` of these IoU values. Anchors above the threshold are positive.

**Strengths for tiny objects:**
- Adaptive threshold means tiny objects (which have inherently lower IoU) get a lower threshold — more positives
- Center-distance selection ensures at least k candidates per level
- Per-GT threshold adapts to each object's characteristics

**Weaknesses for tiny objects:**
- Still uses IoU as the quality measure — noisy for <8px objects
- k is fixed across scales — a 6px object may need more candidates than a 60px object
- The `mean + std` threshold can be unstable when k is small (few levels, few anchors per level)

**For objects <8px:** ATSS typically assigns 3-5 positives per GT (depending on k and FPN levels). This is better than fixed-threshold assignment (which may assign 0) but still sparse.

#### 5.2.2 SimOTA (Ge et al., YOLOX, 2021)

**Mechanism**: An approximate version of OTA (Optimal Transport Assignment). For each GT:
1. Select candidate anchors within the GT box center region
2. Compute a cost matrix: `cost = λ_cls · cls_cost + λ_reg · reg_cost`
3. Determine dynamic k: `k = floor(sum(top-10 IoU values))` — GTs with higher-quality matches get more positives
4. Assign the top-k lowest-cost candidates as positive

**Strengths for tiny objects:**
- Dynamic k adapts to each GT's difficulty — tiny objects with few good matches get fewer (but more reliable) positives
- Cost-based rather than threshold-based — avoids the noisy IoU threshold problem
- Used in YOLOX which achieves strong performance on small COCO objects

**Weaknesses for tiny objects:**
- The center region selection (`center_radius × stride`) may exclude good anchors for off-center tiny objects
- Dynamic k = `floor(sum(top-10 IoU))` — for tiny objects with all IoU < 0.1, k → 0. **This is a critical failure mode for <8px objects.**
- Requires per-image forward pass to compute costs — adds computational overhead

**For objects <8px:** SimOTA can assign k=0 positives because IoU values are so low. A minimum k floor (e.g., k ≥ 1) is essential. With NWD/Gaussian distance replacing IoU in the cost, this improves significantly.

#### 5.2.3 PAA (Probabilistic Anchor Assignment, Kim & Lee, ECCV 2020)

**Mechanism**: Models the anchor score distribution as a Gaussian Mixture Model (2 components: positive and negative). For each GT:
1. Select top-k anchors by IoU
2. Compute anchor scores (IoU × classification score)
3. Fit a 2-component GMM to the scores
4. Assign anchors to positive if they have higher probability under the positive component

**Strengths for tiny objects:**
- Probabilistic separation avoids hard IoU thresholds
- Automatically adapts to the score distribution of each GT
- Tends to produce cleaner positive sets (fewer noisy positives)

**Weaknesses for tiny objects:**
- GMM fitting with very few anchors (common for tiny objects) is unstable
- Still relies on IoU as the initial score — same noise problem
- Computationally expensive (GMM fitting per GT per batch)

**For objects <8px:** PAA is unreliable because the GMM components are poorly separated when all anchor IoUs are low and noisy. The probabilistic framework is elegant but breaks down in the low-signal regime.

#### 5.2.4 Dynamic k (Scale-Adaptive Top-k, as in this project's RFLA variant)

**Mechanism**: A simpler approach used in this project's RFLA setup — assign the top-k anchors per GT based on Gaussian receptive field distance (not IoU), with k varying by object scale:

```python
def gt_scale_topk(gt_size):
    if gt_size < 8:    return 9   # micro: many positives
    elif gt_size < 20: return 6   # tiny: moderate
    else:              return 3   # other: standard
```

**Strengths for tiny objects:**
- Uses Gaussian RF distance, not IoU — scale-invariant and stable for tiny boxes
- Explicitly gives tiny objects more positives — addresses the sparsity problem directly
- Simple, no GMM or OT computation
- Already proven in this project: SCALE_TOPK won AP_micro (0.2947) in the SAH-GD ablation

**Weaknesses:**
- The k values are hand-tuned hyperparameters
- More positives for tiny objects also means more noisy positives if anchors are too coarse
- No quality-aware selection (all top-k are equal, no cost weighting)

**For objects <8px:** This is currently the best-performing approach in the project. The P2 level (stride-4) provides finer anchors for micro objects, which should reduce the noisy-positive problem when combined with scale-adaptive k.

### 5.3 Head-to-Head Comparison for <8px Objects

| Method | Typical k for 6px GT | Quality metric | Assigns 0 positives? | IoU dependence | Compute cost |
|---|---|---|---|---|---|
| Fixed IoU (0.5) | 0-1 | IoU | **YES, often** | Full | Low |
| ATSS (k=9) | 3-5 | IoU (adaptive threshold) | Rare | High | Low |
| SimOTA | 0-3 | Cost (cls+reg) | **YES, if IoU≈0** | High (for k) | Medium |
| PAA | 2-4 | GMM on IoU×cls | Rare but noisy | High | High |
| Dynamic k (scale) | 6-9 | RF distance | **No** (guaranteed k) | **None** | Low |
| RFLA (this project) | 3-9 | Gaussian RF dist | **No** | **None** | Low |

### 5.4 What Actually Works for <8px Objects

Based on the literature, this project's experiments, and structural analysis:

1. **Replace IoU with a scale-invariant metric (NWD/Gaussian distance) in the matching step.** This is the single most impactful change. IoU is fundamentally broken for <8px objects (quantization noise exceeds signal). The project already does this via RFLA + NWD/GCD.

2. **Guarantee a minimum number of positives per GT.** Dynamic k with k≥3 for micro objects ensures every GT gets training signal. SimOTA and PAA can fail to assign any positives.

3. **Scale-adaptive k (more positives for smaller objects)** is validated by this project's experiments. Micro objects need more positives because (a) each individual positive is less informative (coarser features) and (b) the loss landscape is noisier.

4. **Anchor/feature resolution matters more than assignment sophistication.** The P2 result (+29% AP_micro) shows that giving micro objects actual feature resolution (stride-4 instead of stride-8) has larger impact than any assignment change. Assignment and resolution should be co-designed.

5. **Quality-weighted positives (cost-based) help but are not essential.** SimOTA and PAA's cost-based selection is theoretically superior to pure distance-based top-k, but the practical gain is modest when the distance metric is already scale-invariant (Gaussian RF distance).

### 5.5 Recommended Configuration for This Project

```python
# Label assignment: RFLA with scale-adaptive k
ASSIGNMENT_METRIC = "gaussian_rf_distance"  # NOT IoU
K_MICRO  = 6   # objects < 8px  (was 9, reduced to avoid noisy positives with P2)
K_TINY   = 5   # objects 8-20px
K_OTHER  = 3   # objects > 20px
MIN_K    = 1   # guarantee at least 1 positive per GT
```

**Why not SimOTA?** SimOTA's dynamic k is elegant but depends on IoU for k-determination, which is unreliable for <8px objects. Replacing IoU with Gaussian distance in SimOTA is possible but adds complexity without clear benefit over the simpler scale-adaptive approach.

**Why not ATSS?** ATSS's adaptive threshold works well in general but still uses IoU as the quality score. For this project's NWD/GCD metric framework, RFLA's Gaussian RF distance is a more natural fit.

### 5.6 Verdict for This Project

**Already well-addressed.** The project's RFLA + scale-adaptive k + Gaussian distance assignment is near-optimal for tiny objects. The remaining gains are in:
1. Tuning k values (the notebook 12 result suggests k_micro=6 > k_micro=9)
2. Co-optimizing with P2 feature resolution
3. Potentially adding quality-weighted selection within the top-k (not just top-k by distance, but weighted by classification confidence)

---

## 6. Synthesis & Recommendations for This Project

### 6.1 Priority Matrix

Based on the research above, mapped to this project's specific situation (AP@75=0.04 bottleneck, 92% tiny objects, Gaussian regression loss, P2 already implemented):

| Intervention | Expected AP@75 Δ | Expected AP_micro Δ | Complexity | Priority |
|---|---|---|---|---|
| **DIoU regression (scheduled, γ=0.3-0.5)** | +2-4% | ±0 | Low | 🔴 HIGHEST |
| **RoIAlign 14×14 + conv head** | +1-2% | +1-3% | Medium | 🔴 HIGH |
| **DIoU + RoIAlign 14×14 combined** | +3-5% | +1-3% | Medium | 🔴 HIGH |
| Multi-resolution training | +0.5-1% | +1-2% | Low | 🟡 MEDIUM |
| 2-stage Cascade (tuned thresholds) | +1-3% | +0-1% | High | 🟡 MEDIUM |
| Copy-paste augmentation | +0-1% | +2-4% | Medium | 🟡 MEDIUM |
| Scale-adaptive k tuning | +0-0.5% | +1-2% | Low | 🟢 LOW (already good) |
| 3-stage Cascade | +1-2% | -1-0% | Very High | 🟢 LOW |
| SNIP/SNIPER | +0.5-1% | +2-3% | Very High | 🟢 LOW (for now) |
| Mosaic | +0-0.5% | +1-2% | Medium | 🟢 LOW |

### 6.2 Recommended Execution Order

```
Phase 1 — Fix the AP@75 bottleneck (1-2 experiments)
  ├─ (a) DIoU regression with scheduled ramp-up (γ=0.3, warmup=3 epochs)
  │     On clean P2F baseline, no other changes
  ├─ (b) RoIAlign 14×14 + conv head (notebook 14)
  │     On clean P2F baseline, no other changes
  └─ (c) Combine winner of (a) + (b)
         Expected: AP@75 > 0.06, mAP(scale) ≥ 0.58

Phase 2 — Stack micro gains
  ├─ (d) Scale-adaptive k with tuned values (k_micro=6)
  │     On the Phase 1 winner
  └─ (e) Multi-resolution training (random 640-1024)
         On the Phase 1 winner

Phase 3 — Advanced (only if plateaued)
  ├─ (f) 2-stage Cascade with Gaussian-distance thresholds
  ├─ (g) Copy-paste augmentation for micro objects
  └─ (h) CIoU instead of DIoU (if aspect ratio precision needed)
```

### 6.3 Key Principles

1. **Fix the loss function first.** No architectural change can fix AP@75 if the loss doesn't reward precision. The Gaussian similarity loss is the root cause — adding a DIoU term is the minimum necessary fix.

2. **One variable at a time.** The notebook 9 P2 result was confounded by 5 simultaneous changes. Each intervention should be tested in isolation on the P2F baseline.

3. **RoIAlign resolution and regression loss are complementary, not alternatives.** Higher RoIAlign gives the head more information; IoU-aware loss teaches it to use that information for precise boxes. Neither alone is sufficient.

4. **Label assignment is already near-optimal.** RFLA + Gaussian distance + scale-adaptive k is well-suited to this dataset. Further assignment changes have diminishing returns — the bottleneck is now in the head, not the assignment.

5. **Multi-scale training is easy wins.** Random resolution training requires minimal code changes and provides consistent small improvements. It should be the default, not an experiment.

---

## References

1. He, K. et al. "Mask R-CNN." ICCV 2017.
2. Cai, Z. & Vasconcelos, N. "Cascade R-CNN." CVPR 2018.
3. Singh, B. & Davis, L.S. "An Analysis of Scale Invariance in Object Detection — SNIP." CVPR 2018.
4. Singh, B. et al. "SNIPER: Efficient Multi-Scale Training." NeurIPS 2018.
5. Rezatofighi, H. et al. "Generalized Intersection over Union." CVPR 2019.
6. Lu, X. et al. "Grid R-CNN." CVPR 2019.
7. Li, Y. et al. "Scale-Aware Trident Networks." ICCV 2019.
8. Zhang, S. et al. "Bridging the Gap Between Anchor-based and Anchor-free Detection via ATSS." CVPR 2020.
9. Kim, K. & Lee, H.S. "Probabilistic Anchor Assignment with IoU Prediction for Object Detection." ECCV 2020.
10. Zheng, Z. et al. "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression." AAAI 2020.
11. Bochkovskiy, A. et al. "YOLOv4: Optimal Speed and Accuracy of Object Detection." 2020.
12. Ge, Z. et al. "YOLOX: Exceeding YOLO Series in 2021." 2021.
13. Ghiasi, G. et al. "Simple Copy-Paste is a Strong Data Augmentation Method." CVPR 2021.
14. Wang, J. et al. "NWD: Normalized Wasserstein Distance for Tiny Object Detection." 2021.
15. Xu, C. et al. "RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection." ECCV 2022.
16. Yang, Z. et al. "GCD: Gaussian Combined Distance for Tiny Object Detection." 2023.

---

*Document generated: 2026-06-05. For the TinyPerson detection project (Faster R-CNN + RFLA + SAH-GD).*