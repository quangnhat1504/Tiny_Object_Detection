# Bảng Tổng Hợp Đánh Giá Chính Thức Trên 14.018 Ảnh Test AI-TOD-v2

* **Tập dữ liệu**: `D:\paper_a_data\AI-TOD-v2\AI-TOD\images\test` (14.018 ảnh test thực tế, 8 lớp vi mô).
* **Evaluator**: Pinned `aitodpycocotools` theo chuẩn chính thức của tác giả AI-TOD-v2 (Wang et al.).
* **Môi trường**: Huấn luyện độc lập trên cụm Kaggle Tesla T4 GPU (PyTorch AMP) $\to$ Suy luận thực nghiệm trên NVIDIA GeForce RTX 5070 Ti local.

---
## 1. Bảng So Sánh Toàn Diện (Table 2 in Manuscript)

| Phương Pháp (Method) | Tổng Dự Đoán | AP (%) | $\text{AP}_{50}$ (%) | $\text{AP}_{75}$ (%) | $\text{AP}_{vt}$ (%) | $\text{AP}_t$ (%) | $\text{AP}_s$ (%) | $\text{AP}_m$ (%) | $\text{AR}_{1500}$ (%) | oLRP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H-WIoU Unified Proposed (sigma0=8.0px)** | 716,719 | **16.57** | 41.30 | 10.30 | 5.72 | 16.03 | 21.08 | 31.50 | 26.58 | 0.8482 |
| **H-WIoU Proposed (sigma0=8.0px)** | 702,060 | **16.91** | 41.25 | 10.82 | 4.36 | 16.88 | 22.02 | 31.75 | 26.29 | 0.8486 |
| **H-WIoU Ablation (sigma0=6.0px)** | 703,529 | **17.04** | 41.50 | 10.74 | 5.20 | 17.05 | 22.09 | 31.18 | 26.34 | 0.8474 |
| **H-WIoU Ablation (sigma0=10.0px)** | 707,591 | **16.72** | 40.71 | 10.53 | 4.55 | 16.81 | 20.97 | 31.32 | 25.87 | 0.8496 |
| **H-WIoU + Cascade R-CNN** | 702,254 | **16.77** | 40.94 | 10.93 | 4.27 | 16.94 | 21.48 | 31.37 | 26.02 | 0.8494 |
| **Faster R-CNN Baseline (Standard IoU)** | 755,274 | **16.99** | 41.39 | 10.92 | 4.19 | 16.43 | 22.29 | 32.73 | 25.67 | 0.8469 |
| **NWD (NeurIPS 2021)** | 748,450 | **16.79** | 40.62 | 10.91 | 4.54 | 16.50 | 22.19 | 32.43 | 25.62 | 0.8489 |
| **SAFit (AAAI 2024)** | 696,808 | **18.49** | 44.45 | 12.33 | 5.52 | 18.34 | 23.60 | 33.23 | 27.47 | 0.8314 |

---
## 2. Chi Tiết Các Chỉ Số Thu Hồi (Recall & Localization Error Breakdown)

| Phương Pháp | $\text{AR}_{1}$ (%) | $\text{AR}_{100}$ (%) | $\text{AR}_{1500}$ (%) | $\text{AR}_{vt}$ (%) | $\text{AR}_{t}$ (%) | $\text{AR}_{s}$ (%) | $\text{AR}_{m}$ (%) | $\text{oLRP}_{\text{loc}}$ | $\text{oLRP}_{\text{fp}}$ | $\text{oLRP}_{\text{fn}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H-WIoU Unified Proposed (sigma0=8.0px)** | 5.84 | 25.50 | 26.58 | 11.27 | 26.58 | 32.49 | 38.78 | 0.2947 | 0.3772 | 0.5680 |
| **H-WIoU Proposed (sigma0=8.0px)** | 5.70 | 25.12 | 26.29 | 9.59 | 26.72 | 31.39 | 39.12 | 0.2915 | 0.3671 | 0.5878 |
| **H-WIoU Ablation (sigma0=6.0px)** | 5.78 | 25.20 | 26.34 | 9.76 | 26.70 | 31.56 | 38.87 | 0.2899 | 0.3397 | 0.5945 |
| **H-WIoU Ablation (sigma0=10.0px)** | 5.71 | 24.64 | 25.87 | 9.54 | 26.41 | 30.24 | 39.31 | 0.2930 | 0.3669 | 0.5857 |
| **H-WIoU + Cascade R-CNN** | 5.64 | 24.85 | 26.02 | 9.96 | 26.45 | 31.08 | 39.09 | 0.2934 | 0.3585 | 0.5879 |
| **Faster R-CNN Baseline (Standard IoU)** | 5.56 | 24.55 | 25.67 | 10.17 | 26.03 | 30.48 | 38.96 | 0.2915 | 0.3400 | 0.5913 |
| **NWD (NeurIPS 2021)** | 5.52 | 24.51 | 25.62 | 10.75 | 26.14 | 30.52 | 38.62 | 0.2898 | 0.3536 | 0.5993 |
| **SAFit (AAAI 2024)** | 5.84 | 26.27 | 27.47 | 11.14 | 28.21 | 31.22 | 38.78 | 0.2854 | 0.3133 | 0.5573 |

---
## 3. Phân Rã Hiệu Năng Theo Từng Lớp Mục Tiêu (Table 3 in Manuscript - Per-Category AP50 %)

| Phương Pháp | Airplane | Bridge | Storage | Ship | Pool | Vehicle | Person | Windmill | Mean $\text{AP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
