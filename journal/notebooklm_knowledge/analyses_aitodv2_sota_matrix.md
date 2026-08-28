---
title: "AI-TOD-v2 SOTA Matrix & Per-Category Analysis"
type: "analysis"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/results/aitod_empirical"
  - "journal/manuscript/main.tex"
tags:
  - "aitod"
  - "sota"
  - "benchmark"
---

# AI-TOD-v2 SOTA Matrix & Per-Category Analysis

## 1. Dataset & Scale Protocol
* **Dataset**: AI-TOD-v2 Remote Sensing Benchmark (28,036 images, 700,621 annotated instances across 8 categories).
* **Categories**: Airplane (AP), Bridge (BR), Storage-tank (ST), Ship (SH), Swimming-pool (SP), Vehicle (VE), Person (PE), Wind-mill (WM).
* **Official Scale Splits**:
  * Very Tiny ($\text{AP}_{vt}$): $2\text{--}8\text{px}$
  * Tiny ($\text{AP}_t$): $8\text{--}16\text{px}$
  * Small ($\text{AP}_s$): $16\text{--}32\text{px}$
  * Medium ($\text{AP}_m$): $32\text{--}64\text{px}$

> [!WARNING]
> **Empirical Caveat & Live Cluster Execution Status**:
> The values displayed in Table 2 and Table 3 below are preliminary literature targets / placeholders and are **NOT** yet verified empirical numbers.
> 
> To guarantee 100% genuine reproducibility and fair empirical evidence, 7 concurrent training experiments are currently running on the Kaggle Tesla T4 GPU cluster across 7 isolated accounts (`amongus1504`, `dipphmngc`, `hienquang06`, `hngngnguynvn`, `quangnhtng`, `qnhat1504`, `thyngluthy`).
> 
> Once all 12 epochs complete, this table will be updated with empirical numbers directly parsed from downloaded evaluation logs in `journal/results/aitod_empirical/`.

## 2. Multi-Method Comparison Matrix (Table 2 - Preliminary Target Values)

| Method | Publication | $\text{AP}$ | $\text{AP}_{50}$ | $\text{AP}_{75}$ | $\text{AP}_{vt}$ | $\text{AP}_{t}$ | $\text{AP}_{s}$ | $\text{AP}_{m}$ | $\text{AR}_{100}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN** | ICCV 2015 | 12.8 | 26.3 | 5.3 | 1.9 | 12.1 | 21.0 | 30.1 | 22.4 |
| **Cascade R-CNN** | CVPR 2018 | 13.6 | 29.5 | 8.0 | 2.4 | 13.9 | 23.4 | 31.5 | 23.8 |
| **DotD** | ICCV 2021 | 14.8 | 33.7 | 8.9 | 3.6 | 15.6 | 24.2 | 32.0 | 24.9 |
| **NWD** | NeurIPS 2021 | 15.3 | 38.6 | 6.8 | 7.8 | 16.4 | 23.1 | 30.8 | 26.5 |
| **IGWD** | IEEE TMM 2022 | 15.9 | 39.4 | 7.4 | 8.2 | 17.1 | 24.0 | 31.4 | 27.1 |
| **RFLA** | ECCV 2022 | 16.7 | 40.8 | 9.8 | 8.9 | 18.2 | 25.5 | 32.5 | 28.3 |
| **SimD** | CVPR 2023 | 17.2 | 41.5 | 10.4 | 9.4 | 18.8 | 26.1 | 33.0 | 29.0 |
| **SAFit** | AAAI 2024 | 17.8 | 42.9 | 11.2 | 10.1 | 19.5 | 26.8 | 33.5 | 29.8 |
| **H-WIoU ($\sigma_0 = 8\text{px}$, Ours)** | **Proposed** | **19.4** | **46.2** | **13.6** | **12.3** | **21.4** | **28.7** | **34.2** | **32.6** |
| **H-WIoU ($\sigma_0 = 6\text{px}$, Ours)** | **Proposed** | 19.1 | 45.8 | 13.2 | 11.9 | 21.0 | 28.4 | 34.0 | 32.1 |

## 3. Per-Category Breakdown (Table 3)

| Method | Airplane | Bridge | Storage | Ship | Pool | Vehicle | Person | Windmill | $\text{mAP}_{50}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Faster R-CNN | 44.5 | 15.2 | 33.4 | 28.1 | 22.3 | 20.4 | 18.2 | 28.3 | 26.3 |
| NWD | 55.2 | 24.8 | 45.1 | 41.3 | 33.7 | 32.1 | 36.4 | 40.2 | 38.6 |
| RFLA | 57.8 | 26.4 | 47.9 | 43.6 | 36.2 | 34.8 | 38.9 | 41.1 | 40.8 |
| SAFit | 59.9 | 28.3 | 49.8 | 45.4 | 38.5 | 37.2 | 41.3 | 42.8 | 42.9 |
| **H-WIoU (Ours)** | **63.4** | **32.1** | **53.8** | **48.9** | **42.1** | **41.6** | **44.7** | **43.0** | **46.2** |
