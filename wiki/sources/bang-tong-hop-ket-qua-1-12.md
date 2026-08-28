---
title: Bảng Tổng Hợp Kết Quả Notebooks 1-12
type: source
created: 2026-06-05
updated: 2026-06-05
sources: [raw/bang_tong_hop_ket_qua_1_12]
tags: [experiment-results, summary, vietnamese]
---

## Bảng Tổng Hợp Kết Quả Notebooks 1-12

### Tổng Quan

Đây là bảng tổng hợp kết quả từ 12 notebooks thực nghiệm trên TinyPerson dataset với kiến trúc Faster R-CNN + RFLA. Các thí nghiệm được chia thành 3 giai đoạn:

- **Notebooks 1-4**: Thử nghiệm các metric cơ bản (NWD, GCD, IGWD, ALW)
- **Notebooks 5-8**: Thử nghiệm các biến thể SAH-GD (Scale-Adaptive Hybrid Gaussian Distance)
- **Notebooks 9-12**: Thử nghiệm các cải tiến với P2 feature level

### Bảng Kết Quả Chi Tiết

| # | Notebook | Method | mAP(scale) | mAP@50 | AP_micro | AP_tiny | AP_small | AP_large | COCO AP@50:75 | COCO AP@75 | AR@100 | Det/img |
|---|----------|--------|------------|--------|----------|---------|----------|----------|---------------|------------|--------|---------|
| 1 | 1_nwd.ipynb | **NWD** | 0.3967 | 0.2039 | **0.2652** | 0.4741 | 0.3603 | 0.2597 | 0.1162 | 0.0376 | 0.3852 | 214.68 |
| 2 | 2_gcd.ipynb | **GCD** | 0.5522 | 0.3483 | 0.2582 | 0.5437 | 0.6370 | 0.7455 | 0.1821 | 0.0415 | 0.3939 | 119.38 |
| 3 | 3_igwd.ipynb | **IGWD** | 0.5187 | 0.3294 | 0.1928 | 0.5084 | 0.6156 | 0.7075 | 0.1639 | 0.0312 | 0.3650 | 133.92 |
| 4 | 4_alw.ipynb | **ALW** | 0.1822 | 0.1256 | 0.1029 | 0.2190 | 0.1656 | 0.1744 | 0.0652 | 0.0145 | 0.3589 | 599.20 |
| 5 | 5_adaptive_nwd.ipynb | **ADAPTIVE_NWD** | 0.5671 | 0.3354 | 0.2395 | 0.5617 | 0.6610 | 0.7296 | 0.1676 | 0.0345 | 0.3551 | 94.63 |
| 6 | 6_hard_switch_nwd_gcd.ipynb | **HARD_SWITCH_NWD_GCD** ⭐ | **0.5770** | 0.3517 | 0.2776 | 0.5721 | 0.6600 | **0.7620** | 0.1831 | 0.0428 | 0.3818 | 100.08 |
| 7 | 7_sah_gd_soft_blend.ipynb | **SAH_GD_SOFT_BLEND** | 0.5752 | 0.3537 | 0.2758 | 0.5686 | **0.6604** | 0.7559 | 0.1848 | 0.0447 | 0.3827 | 100.68 |
| 8 | 8_sah_gd_scale_topk.ipynb | **SAH_GD_SCALE_TOPK** 🎯 | 0.5768 | **0.3592** | 0.2947 | **0.5810** | 0.6487 | 0.7059 | **0.1888** | **0.0453** | **0.3869** | 101.46 |
| 9 | 9_hard_switch_p2.ipynb | **HARD_SWITCH_P2F** | 0.4292 | 0.2358 | 0.2717 | 0.4695 | 0.4449 | 0.2591 | 0.0971 | 0.0130 | 0.2815 | 79.02 |
| 10 | 10_dual_reg_p2.ipynb | **HARD_SWITCH_P2_DUAL** | 0.4516 | 0.2452 | 0.2806 | 0.4901 | 0.4734 | 0.2852 | 0.1032 | 0.0129 | 0.2886 | 72.54 |
| 11 | 11_scale_topk_p2.ipynb | **SCALE_TOPK_P2** | 0.4522 | 0.2493 | 0.2821 | 0.4971 | 0.4666 | 0.2823 | 0.1039 | 0.0121 | 0.2956 | 75.02 |
| 12 | 12_hard_switch_p2_topk_dual.ipynb | **HARD_SWITCH_P2_TOPK_DUAL** 💎 | 0.4724 | 0.2597 | **0.3151** | 0.5151 | 0.4863 | 0.2903 | 0.1099 | 0.0145 | 0.2959 | 70.45 |

**Chú thích:**
- ⭐ = Tốt nhất về tổng thể (mAP)
- 🎯 = Tốt nhất cho micro/tiny trong SAH-GD
- 💎 = AP_micro cao nhất toàn bộ (0.3151)
- Giá trị in đậm = Kết quả tốt nhất trong cột đó
- COCO AP@50:75 = trung bình COCO AP trên các ngưỡng IoU 0.50-0.75 trong notebook

### Bảng Metrics Sử Dụng

| Metric trong bảng | Nhóm | Cách tính / ý nghĩa | Ghi chú |
|---|---|---|---|
| `mAP(scale)` | Custom | Weighted average của `AP_micro`, `AP_tiny`, `AP_small`, `AP_large`, trọng số theo số lượng GT trong từng scale-bin. | Đây là primary metric tự thiết kế để TinyPerson không bị COCO size-bin làm mờ các vật thể rất nhỏ. |
| `mAP@50` | COCO-style | AP tại IoU threshold `0.50`, tính bằng `torchmetrics.detection.MeanAveragePrecision`. | Dễ đạt hơn AP@75; đo khả năng detect đúng tương đối lỏng. |
| `AP_micro` | Custom | AP chỉ trên GT có `sqrt(area)` trong khoảng `0 <= s < 6 px`; TP nếu IoU đạt ngưỡng custom của bin. | COCO không có nhóm `micro`; đây là bin quan trọng nhất cho vật thể cực nhỏ. |
| `AP_tiny` | Custom | AP chỉ trên GT có `6 <= sqrt(area) < 16 px`; TP nếu IoU đạt ngưỡng custom của bin. | COCO không có nhóm `tiny`; dùng để tách rõ vùng nhỏ hơn COCO-small. |
| `AP_small` | Custom | AP chỉ trên GT có `16 <= sqrt(area) < 64 px`; TP nếu IoU đạt ngưỡng custom của bin. | Không phải `COCO AP_small`; tên giống nhưng định nghĩa khác. |
| `AP_large` | Custom | AP chỉ trên GT có `sqrt(area) >= 64 px`; TP nếu IoU đạt ngưỡng custom của bin. | Không phải `COCO AP_large`; trong TinyPerson bin này ít mẫu nên variance cao. |
| `COCO AP@50:75` | COCO-style tùy chỉnh | AP trung bình trên các ngưỡng IoU `0.50, 0.55, 0.60, 0.65, 0.70, 0.75`. | Không phải COCO main AP chuẩn `0.50:0.95`; đây là truncated COCO AP. |
| `COCO AP@75` | COCO | AP tại IoU threshold `0.75`. | Đo localization nghiêm ngặt; đang là bottleneck chính. |
| `AR@100` | COCO-style | Average Recall với tối đa 100 detections/image, lấy từ `mar_100`. | Đo recall khi giới hạn số detection. |
| `Det/img` | Custom diagnostic | Tổng số detections trên test/inference chia cho số ảnh. | Không phải accuracy metric; dùng để phát hiện over-detection/duplicate boxes. |

**Chi tiết custom scale AP:** trong notebook, hàm `_compute_scale_ap` chia GT theo `s = sqrt(width * height)`: `micro=(0,6)`, `tiny=(6,16)`, `small=(16,64)`, `large=(64,+∞)`. Mỗi bin dùng IoU threshold riêng: `micro=0.25`, `tiny=0.25`, `small=0.35`, `large=0.50`. AP được tính bằng precision-recall 101 điểm giống tinh thần AP, nhưng phần chia bin và threshold là custom cho tiny-object detection.

### Phân Tích Tiến Triển

#### Giai Đoạn 1: Baseline Metrics (Notebooks 1-4)

**Kết quả nổi bật:**
- **GCD (NB2)** là metric tốt nhất: mAP=0.5522, cân bằng tốt
- **NWD (NB1)** tốt nhất cho micro objects: AP_micro=0.2652
- **ALW (NB4)** thất bại hoàn toàn: mAP=0.1822, quá nhiều detections (599/img)
- **IGWD (NB3)** trung bình: mAP=0.5187

**Vấn đề chung:** Tất cả có COCO AP@75 rất thấp (0.0312-0.0415) → localization yếu

#### Giai Đoạn 2: SAH-GD Variants (Notebooks 5-8)

**Tiến bộ đạt được:**
- Tất cả SAH-GD variants đều vượt baseline GCD
- **HARD_SWITCH_NWD_GCD (NB6)**: +4.5% mAP so với GCD (0.5770 vs 0.5522)
- **SCALE_TOPK (NB8)**: +14.1% AP_micro so với GCD (0.2947 vs 0.2582)
- Giảm duplicate detections: 94-101 det/img vs 119 (GCD baseline)
- Cải thiện localization: AP@75 lên 0.0453 (SCALE_TOPK)

**So sánh các SAH-GD variants:**
1. **HARD_SWITCH (NB6)** ⭐: Tốt nhất tổng thể, đơn giản, nhanh
2. **SCALE_TOPK (NB8)** 🎯: Tốt nhất cho micro/tiny, AP@75 cao nhất
3. **SOFT_BLEND (NB7)**: Không có lợi thế rõ ràng, chậm nhất
4. **ADAPTIVE_NWD (NB5)**: Nhanh nhất nhưng AP_micro thấp nhất

#### Giai Đoạn 3: P2 Feature Level (Notebooks 9-12)

**Kết quả:**
- **Tất cả notebooks P2 đều giảm hiệu suất** so với SAH-GD (notebooks 5-8)
- mAP giảm từ ~0.57 xuống ~0.42-0.47
- Giảm detections xuống 70-79/img (tốt) nhưng mất precision
- AP_large giảm mạnh: từ 0.76 xuống 0.26-0.29

**P2 variants so sánh (theo mAP):**
1. **HARD_SWITCH_P2_TOPK_DUAL (NB12)**: mAP=0.4724, AP_micro=0.3151 (tốt nhất)
2. **SCALE_TOPK_P2 (NB11)**: mAP=0.4522, AP_micro=0.2821
3. **HARD_SWITCH_P2_DUAL (NB10)**: mAP=0.4516, AP_micro=0.2806
4. **HARD_SWITCH_P2F (NB9)**: mAP=0.4292, AP_micro=0.2717 (baseline P2)

**Nhận xét:**
- NB9 là P2 baseline, các notebooks 10-12 thử cải tiến nhưng chỉ NB12 tăng đáng kể
- Tất cả P2 variants vẫn kém xa SAH-GD (mAP 0.42-0.47 vs 0.57)
- AP_micro của NB12 (0.3151) là cao nhất trong toàn bộ 12 notebooks
- Det/img giảm tốt (70-79) so với SAH-GD (94-101) nhưng AP_large sụt mạnh

### Top 3 Configurations Tổng Kết

| Hạng | Notebook | Method | mAP | AP_micro | AP_tiny | Đặc điểm |
|------|----------|--------|-----|----------|---------|----------|
| 🥇 | 6 | HARD_SWITCH_NWD_GCD | 0.5770 | 0.2776 | 0.5721 | Tốt nhất tổng thể, cân bằng, đơn giản |
| 🥈 | 8 | SAH_GD_SCALE_TOPK | 0.5768 | 0.2947 | 0.5810 | Tốt nhất cho micro/tiny, AP@75 cao nhất |
| 🥉 | 7 | SAH_GD_SOFT_BLEND | 0.5752 | 0.2758 | 0.5686 | Tốt cho small objects |

### Khuyến Nghị

**Dùng ngay:**
- **HARD_SWITCH_NWD_GCD (NB6)** cho ứng dụng cần mAP tổng thể cao
- **SAH_GD_SCALE_TOPK (NB8)** cho ứng dụng tập trung vào micro/tiny objects

**Không nên dùng:**
- ALW (NB4): Quá yếu
- Các P2 variants (NB10-12): Hiệu suất kém hơn SAH-GD

**Hướng cải thiện tiếp theo:**
1. Giữ HARD_SWITCH (NB6) làm baseline chính
2. Cải thiện architecture (không chỉ metric):
   - Tối ưu P2 feature resolution
   - Retune anchors theo EDA bins
   - Cải thiện post-processing/NMS
3. Kết hợp ưu điểm SCALE_TOPK cho micro objects

### Bottlenecks Chính

1. **Localization yếu**: AP@75 vẫn thấp (0.0121-0.0453) → cần cải thiện feature resolution
2. **P2 implementation chưa tốt**: Notebooks 10-12 giảm performance thay vì cải thiện
3. **Large objects trong P2**: AP_large giảm mạnh khi thêm P2 → cần balance better

### Dữ Liệu Tham Khảo

- Dataset: TinyPerson (70,702 objects, 1,570 images)
- 79.5% objects có sqrt(area) < 20px
- 92.1% objects có sqrt(area) < 32px
- Architecture: Faster R-CNN + RFLA
- Training: 12 epochs cho SAH-GD variants

### Related Pages

- [[SAH-GD Hybrid Metrics Comparison]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[Tiny Object Metrics Comparison Filled]]
- [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]]
