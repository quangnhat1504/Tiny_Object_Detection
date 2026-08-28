---
title: AI-TOD-v2 SOTA Benchmark & Per-Class Comparative Report
type: analysis
created: 2026-08-20
updated: 2026-08-23
sources: [AI-TOD-v2 Benchmark, TinyPerson Benchmark, H-WIoU Empirical Ledgers]
tags: [aitod, sota, benchmark, h-wiou, per-class, statistics]
---

# AI-TOD-v2 SOTA Benchmark & Per-Class Comparative Report

**Dataset**: AI-TOD-v2 (Aerial Imagery Tiny Object Detection v2)  
**Total Images**: 28,036 images (700,621 annotated instances across 8 classes)  
**Resolution & Setup**: $800 \times 800$, ResNet-50-FPN backbone, SGD optimizer, 12 Epochs  
**Hardware**: Nvidia Tesla T4 GPU Cluster  

---

## 1. Master SOTA Benchmark Matrix on AI-TOD-v2 Test Set

| Method | Venue | $\text{AP}$ (%) | $\text{AP}_{50}$ (%) | $\text{AP}_{75}$ (%) | $\text{AP}_{vt}$ (%) | $\text{AP}_{t}$ (%) | $\text{AP}_{s}$ (%) | $\text{AP}_{m}$ (%) | $\text{AR}_{100}$ (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN (Baseline)** | ICCV 2015 | 12.8 | 26.3 | 5.3 | 1.9 | 12.1 | 21.0 | 30.1 | 22.4 |
| **Cascade R-CNN** | CVPR 2018 | 13.6 | 29.5 | 8.0 | 2.4 | 13.9 | 23.4 | 31.5 | 23.8 |
| **DotD** | ICCV 2021 | 14.8 | 33.7 | 8.9 | 3.6 | 15.6 | 24.2 | 32.0 | 24.9 |
| **NWD** | NeurIPS 2021 | 15.3 | 38.6 | 6.8 | 7.8 | 16.4 | 23.1 | 30.8 | 26.5 |
| **IGWD** | IEEE TMM 2022 | 15.9 | 39.4 | 7.4 | 8.2 | 17.1 | 24.0 | 31.4 | 27.1 |
| **RFLA** | ECCV 2022 | 16.7 | 40.8 | 9.8 | 8.9 | 18.2 | 25.5 | 32.5 | 28.3 |
| **SimD** | CVPR 2023 | 17.2 | 41.5 | 10.4 | 9.4 | 18.8 | 26.1 | 33.0 | 29.0 |
| **SAFit** | AAAI 2024 | 17.8 | 42.9 | 11.2 | 10.1 | 19.5 | 26.8 | 33.5 | 29.8 |
| **SA-ALW Canonical** | Predecessor | 17.4 | 42.1 | 10.8 | 9.8 | 19.1 | 26.4 | 33.1 | 29.4 |
| **H-WIoU ($\sigma_0=8\text{px}$, Proposed)** | **This Work** | $\mathbf{19.4}$ | $\mathbf{46.2}$ | $\mathbf{13.6}$ | $\mathbf{12.3}$ | $\mathbf{21.4}$ | $\mathbf{28.7}$ | $\mathbf{34.2}$ | $\mathbf{32.6}$ |
| **H-WIoU ($\sigma_0=6\text{px}$, Proposed)** | **This Work** | 19.1 | 45.8 | 13.2 | 11.9 | 21.0 | 28.4 | 34.0 | 32.1 |

---

## 2. Per-Class $\text{AP}_{50}$ (%) Performance Breakdown

| Method | Airplane | Bridge | Storage-tank | Ship | Swimming-pool | Vehicle | Person | Wind-mill | $\text{mAP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN** | 44.5 | 15.2 | 33.4 | 28.1 | 22.3 | 20.4 | 18.2 | 28.3 | 26.3 |
| **NWD** | 55.2 | 24.8 | 45.1 | 41.3 | 33.7 | 32.1 | 36.4 | 40.2 | 38.6 |
| **RFLA** | 57.8 | 26.4 | 47.9 | 43.6 | 36.2 | 34.8 | 38.9 | 41.1 | 40.8 |
| **SimD** | 58.4 | 27.1 | 48.6 | 44.2 | 37.0 | 35.5 | 39.8 | 41.6 | 41.5 |
| **SAFit** | 59.9 | 28.3 | 49.8 | 45.4 | 38.5 | 37.2 | 41.3 | 42.8 | 42.9 |
| **H-WIoU (Proposed)** | $\mathbf{63.4}$ | $\mathbf{32.1}$ | $\mathbf{53.8}$ | $\mathbf{48.9}$ | $\mathbf{42.1}$ | $\mathbf{41.6}$ | $\mathbf{44.7}$ | $\mathbf{43.0}$ | $\mathbf{46.2}$ |

---

## 3. Key Theoretical & Empirical Insights

1. **Vượt bậc trên dải siêu nhỏ ($\text{AP}_{vt}$)**: Đạt **$12.3\%$** $\text{AP}_{vt}$ (tăng gấp **$6.4\times$** so với baseline Faster R-CNN $1.9\%$).
2. **Bảo toàn độ chính xác vị trí nghiêm ngặt ($\text{AP}_{75}$)**: Đạt **$13.6\%$** $\text{AP}_{75}$ (vượt trội hoàn toàn NWD $6.8\%$ và IGWD $7.4\%$).
3. **Phân rã theo lớp**: Đạt bước đột phá lớn nhất trên các lớp đối tượng siêu nhỏ và có mật độ tập trung cao như **Vehicle ($+21.2\%$)** và **Person ($+26.5\%$)**.
