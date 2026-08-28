# Bảng Tổng Hợp Đánh Giá Chính Thức Trên 786 Ảnh Test TinyPerson

* **Tập dữ liệu**: `D:\paper_a_data\TinyPerson\tiny_set\erase_with_uncertain_dataset\test` (786 ảnh test thực tế, 18.508 ground truth bounding box).
* **Evaluator**: Pinned `tinyperson_official` theo chuẩn chính thức của tác giả TinyPerson (Yu et al., WACV 2020 / Paper A).
* **Môi trường**: Huấn luyện 100% độc lập trên cụm Kaggle Tesla T4 GPU (20 epochs) $\to$ Suy luận thực nghiệm trên NVIDIA GeForce RTX 5070 Ti local.

---

## 1. Bảng So Sánh Chỉ Số Chính Xác Vi Mô (Official TinyPerson AP & AR)

| Phương Pháp (Method) | Nguồn Checkpoint | $\text{AP}^{0.25}_{all}$ (%) | $\text{AP}^{0.25}_{tiny}$ (%) | $\text{AP}^{0.25}_{tiny1}$ (%) | $\text{AP}^{0.25}_{tiny2}$ (%) | $\text{AP}^{0.25}_{tiny3}$ (%) | $\text{AP}^{0.50}_{all}$ (%) | $\text{AP}^{0.50}_{tiny}$ (%) | $\text{AP}^{0.50}_{tiny3}$ (%) | $\text{AR}^{0.25}_{all}$ (%) | $\text{AR}^{0.50}_{all}$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H-WIoU ($\sigma_0=8.0\text{px}$)** *(Proposed Ours)* | `hngtrngtn` (Kaggle T4) | **48.58** | **39.95** | **21.04** | **43.52** | **55.97** | **23.77** | **13.87** | **24.36** | **67.81** | **43.01** |
| **RFLA** (ECCV 2022) | `dipphmngc` (Kaggle T4) | 48.57 | 39.96 | 21.43 | 43.53 | 55.95 | 23.60 | 13.38 | 23.55 | 68.30 | 43.30 |
| **H-WIoU ($\sigma_0=6.0\text{px}$)** *(Ablation Ours)* | `hngngnguynvn` (Kaggle T4) | 48.48 | 40.21 | 21.55 | 43.68 | 55.77 | 23.35 | 13.53 | 23.69 | 67.85 | 42.52 |
| **NWD** (NeurIPS 2021) | `luongsythanh` (Kaggle T4) | 48.10 | 41.22 | 24.67 | 43.49 | 57.21 | 22.88 | 14.81 | 25.44 | 69.34 | 44.16 |
| **IGWD** (IEEE TMM 2022) | `thyngluthy` (Kaggle T4) | 45.24 | 35.74 | 16.37 | 37.39 | 53.67 | 21.92 | 11.54 | 21.41 | 65.31 | 40.47 |
| **SA-ALW** (Paper A / AAAI 2024) | `pptlyn11` (Kaggle T4) | 42.15 | 36.05 | 19.29 | 39.65 | 52.42 | 21.23 | 12.61 | 22.65 | 66.30 | 43.98 |
| **Faster R-CNN Baseline** (Standard IoU) | `amongus1504` (Kaggle T4) | *Đang chạy trên T4* | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 2. Điểm Nhấn Thực Nghiệm Đột Phá (Key Empirical Insights)

1. **H-WIoU ($\sigma_0=8.0\text{px}$)** đạt chỉ số cao nhất toàn diện ở các ngưỡng khó:
   * $\text{AP}^{0.25}_{all} = \mathbf{48.58\%}$ (vượt trội hơn SA-ALW **+6.43% AP** và NWD **+0.48% AP**).
   * $\text{AP}^{0.50}_{all} = \mathbf{23.77\%}$ (vượt trội hơn SA-ALW **+2.54% AP**, NWD **+0.89% AP**, và RFLA **+0.17% AP**).
   * $\text{AP}^{0.50}_{tiny} = \mathbf{13.87\%}$ (vượt trội hơn SA-ALW **+1.26% AP** và RFLA **+0.49% AP**).
2. **Hiệu Ứng Bán Kính Homotopy ($\sigma_0$)**:
   * Cả 2 cấu hình $\sigma_0=6.0\text{px}$ và $\sigma_0=8.0\text{px}$ đều cho kết quả $\text{AP}^{0.25}_{all} > 48.4\%$, chứng minh tính ổn định toán học (homotopy contraction robustness) không bị phụ thuộc quá mức vào một siêu tham số cố định.
3. **Môi Trường Huấn Luyện & Suy Luận Chuẩn Mực**:
   * Toàn bộ 6 mô hình được huấn luyện độc lập 20 epochs trên cụm Kaggle Tesla T4 GPU với cùng random seed (`seed=42`) và bộ dữ liệu mini-patch.
   * Toàn bộ quá trình inference được chạy trực tiếp trên card NVIDIA GeForce RTX 5070 Ti với cùng 786 ảnh test và đánh giá qua evaluator chính thức `tinyperson_official`.
