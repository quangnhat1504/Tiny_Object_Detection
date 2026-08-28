---
title: "Journal Manuscript Architecture & Publication Blueprint"
type: "synthesis"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
  - "journal/manuscript/main.pdf"
tags:
  - "manuscript"
  - "latex"
  - "publication"
---

# Journal Manuscript Architecture & Publication Blueprint

## 1. Document Overview
* **Target Format**: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) / IEEE Transactions on Image Processing (TIP).
* **Length**: 8 full pages, double-column format.
* **Master Source File**: [`journal/manuscript/main.tex`](file:///c:/Users/ADMIN/_Project/tiny-object-detection/journal/manuscript/main.tex)
* **Compiled Artifact**: [`journal/manuscript/main.pdf`](file:///c:/Users/ADMIN/_Project/tiny-object-detection/journal/manuscript/main.pdf)

### Author Roster
1. **Lê Hồ Anh Duy** (ID: DE200171) — Email: `lehoanhduy5426@gmail.com` — Tel: (+84) 898-896-962
2. **Đặng Quang Nhật** (ID: DE200497) — Email: `dangquangnhat1504@gmail.com` — Tel: (+84) 377-231-436
3. **Phạm Minh Tiến** (ID: DE191091) — Email: `taxaceae.forwork@gmail.com` — Tel: (+84) 968-338-702
*Affiliation: Department of Artificial Intelligence & Computer Science, FPT University*

## 2. Structural Section Mapping

| Section | Title | Content Focus | Embedded Assets |
| :---: | :--- | :--- | :--- |
| **1** | **Introduction** | Dilemma of discrete IoU vs Gaussian blurring; summary of 5 principal contributions. | Equation (1) |
| **2** | **Related Work** | Bounding box metrics, Wasserstein & distribution metrics, Label assignment strategies. | Literature review citations |
| **3** | **Mathematical Theory of H-WIoU** | Gaussian embedding, normalized Wasserstein distance, scale-homotopy operator, Proposition 1, Theorem 1 & proof. | Figure 1 (Homotopy Foundations) |
| **4** | **H-WIoU Detection Framework** | Stage 1 Homotopy Label Assignment in RPN; Stage 2 Bounded Homotopy Box Loss in RoI Head. | Figure 5 (PaperBanana Architecture Diagram) |
| **5** | **Experimental Setup** | Datasets (TinyPerson, AI-TOD-v2), evaluation metrics, Fair-20 protocol details. | Implementation parameters |
| **6** | **Main Experimental Results** | TinyPerson benchmark (Table 1), AI-TOD-v2 benchmark (Table 2), Class breakdown (Table 3). | Figure 2 (Multi-Metric Radar Chart), Tables 1-3 |
| **7** | **Ablation Studies** | Scale sensitivity ($\sigma_0$), transition functional forms ($\gamma$), module placement isolation. | Figure 3 (Ablation Landscape) |
| **8** | **Statistical Significance Analysis** | 16-fold paired Student's $t$-test, Wilcoxon signed-rank test, $N=10,000$ bootstrap CIs. | Statistical CI intervals |
| **9** | **Computational Efficiency** | Parameters, GPU memory, inference FPS ($> 54\text{ FPS}$). | Table 4 (Latency benchmark) |
| **10** | **Qualitative Analysis** | Visual detection comparisons across marine and aerial scenes. | Figure 4 (Qualitative Gallery) |
| **11** | **Conclusion** | Synthesis of findings, impact on tiny object detection, future directions. | Final remarks |

## 3. Publication Asset Registry
* `Figure 1`: `journal/manuscript/figures/fig1_homotopy_theory.pdf`
* `Figure 2`: `journal/manuscript/figures/fig2_multimetric_radar.pdf`
* `Figure 3`: `journal/manuscript/figures/fig5_pipeline_architecture.pdf` (Rendered as 2-column pipeline banner)
* `Figure 4`: `journal/manuscript/figures/fig3_ablation_landscape.pdf`
* `Figure 5`: `journal/manuscript/figures/fig4_qualitative_detections.pdf`
