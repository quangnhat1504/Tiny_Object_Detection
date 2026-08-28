# Phase 0 Report — Data Preparation & Statistical Analysis

> Generated: 2026-07-01
> Source data: `eda/instances.csv`, `eda/summary.json`, `data/patches/`

## 1. Dataset Overview (from existing EDA)

| Metric | Value |
|--------|-------|
| Total images | 1,570 |
| Total instances | 70,702 |
| Classes | dry-person (65.8%), wet-swimmer (34.2%) |
| Class imbalance | 1.9:1 (dry:wet) |
| Background images | 3 (all in train) |
| Average density | 45.03 objects/image (max 730) |

### Split sizes

| Split | Images | Objects | obj/img (avg) | obj/img (max) |
|-------|-------:|--------:|:-------------:|:-------------:|
| train | 1,374 | 63,471 | 46.2 | 730 |
| valid | 131 | 4,382 | 33.5 | 271 |
| test | 65 | 2,849 | 43.8 | 335 |

## 2. Scale Distribution (sqrt-area in pixels)

### Full dataset percentiles

| Percentile | Full | dry-person | wet-swimmer |
|:----------:|:----:|:----------:|:-----------:|
| P5 | 4.71 | 4.83 | 4.53 |
| **P10** | **5.59** | **5.74** | **5.32** |
| P25 | 7.70 | 7.79 | 7.53 |
| P50 | 11.50 | 11.30 | 11.92 |
| P75 | 17.92 | 17.17 | 19.40 |
| **P90** | **28.67** | **27.04** | **32.01** |
| P95 | 38.85 | 36.41 | 43.85 |

### Implication for SA-ALW

- **s_min = P10 = 5.6** (dùng cho scale_adaptive_beta và scale_adaptive_pos_weight)
- **s_max = P90 = 28.7**
- Không cần s_min/s_max riêng theo class — chênh lệch giữa dry-person và wet-swimmer không đủ lớn để justify 2 bộ tham số

### Hypothesis H0.1 — BÁC BỎ

Giả thuyết "sea_person (wet-swimmer) nhỏ hơn dry_person do ở xa" là **sai**:
- wet-swimmer P50 = 11.92 > dry-person P50 = 11.30
- wet-swimmer mean = 16.95 > dry-person mean = 15.00
- wet-swimmer actually slightly larger on average

## 3. Size Bins

| Bin (px) | Count | % |
|:--------:|:-----:|:-:|
| (0, 8) | 19,196 | 27.1% |
| [8, 12) | 18,040 | 25.5% |
| [12, 20) | 18,978 | 26.8% |
| [20, 32) | 8,937 | 12.6% |
| [32, 96) | 5,050 | 7.1% |
| >=96 | 501 | 0.7% |

Tiny (<20px): **79.5%** | Small (<32px): **92.1%**

## 4. Aspect Ratio Analysis

| Metric | dry-person | wet-swimmer |
|--------|:----------:|:-----------:|
| Median aspect | 0.531 | 0.881 |
| P25 | 0.418 | 0.644 |
| P75 | 0.706 | 1.187 |
| Flat (>2) | 1.0% | 4.0% |
| Tall (<0.5) | 43.6% | 10.2% |

### Key finding
- **dry-person** là class có aspect ratio cực đoan hơn (43.6% tall — người đứng)
- **wet-swimmer** khá vuông (median 0.881) — không dài/ẹt như giả thuyết ban đầu
- Anisotropic normalization của ALW sẽ có lợi hơn cho **dry-person**, không phải wet-swimmer

## 5. Annotation Quality

| Criterion | Count | % |
|-----------|:-----:|:-:|
| Boxes w<3 or h<3 | 1,113 | 1.6% |
| - dry-person | 932 | (2.0% of dry) |
| - wet-swimmer | 181 | (0.7% of wet) |

Kết luận: tỷ lệ nhiễu annotation thấp (1.6%), không cần làm sạch đặc biệt.

## 6. Stratified Split Check

| Metric | train | valid | test |
|--------|:----:|:-----:|:----:|
| dry:wet ratio | 1.96 | 1.61 | 1.63 |
| sqrt_area median | 11.5 | 13.3 | 10.7 |
| Tiny (<20px) | 79.9% | 70.0% | 85.4% |
| Small (<32px) | 92.4% | 88.0% | 93.8% |
| P10 | 5.6 | 5.3 | 4.9 |
| P90 | 28.3 | 34.9 | 24.0 |

Lưu ý: valid set có object to hơn trung bình (P50=13.3 vs 11.5), test set có object nhỏ hơn (P50=10.7). Điều này có thể ảnh hưởng nhẹ đến việc đọc kết quả validation.

## 7. Patch Training Data Generation

### Script
- `scripts/generate_patches.py` — sinh patch từ annotation (context_ratio=1.5, patch_size=512)
- Chạy: `python scripts/generate_patches.py [train|valid]`

### Output statistics

| Split | Images | Patches | Patches/img | Objects in patches |
|:-----:|:------:|:-------:|:-----------:|:------------------:|
| train | 1,371 | 9,863 | 7.2 | ~355k |
| valid | 131 | 854 | 6.5 | ~23k |
| **Total** | **1,502** | **10,717** | 7.1 | **~378k** |

### Location
- `data/patches/{train,valid}/images/` — patch JPEGs (512×512)
- `data/patches/{train,valid}/labels/` — YOLO bbox labels (class xc yc w h)
- `data/patches/meta.json` — mapping từ ảnh gốc → patches

## 8. Configuration Values for Subsequent Phases

| Parameter | Value | Source |
|-----------|:-----:|--------|
| s_min (SA-β, SA-pos) | 5.6 | P10 of sqrt-area |
| s_max (SA-β, SA-pos) | 28.7 | P90 of sqrt-area |
| β_min | 8.0 | IGWD paper default |
| β_max | 10.0 | IGWD paper (AP_vt optimal) |
| log_clamp | 3.0 | Initial, needs ablation (H2.4) |
| pos_weight_min | 1.0 | Default |
| pos_weight_max | 1.5 | Initial, needs ablation (H2.3) |
| reliability_thr | P25 (adaptive) | From dataset, clipped [4, 24] |
| context_ratio | 1.5 | Initial, tune in Phase 4 |
| patch_size | 512 | Initial, tune in Phase 4 |
