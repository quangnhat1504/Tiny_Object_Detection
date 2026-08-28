---
title: AI-TOD-v2 Full Empirical Benchmark Matrix
type: analysis
created: 2026-08-23
updated: 2026-08-23
sources: []
tags: [aitod, empirical-benchmark, kaggle-cluster, sota]
---

# AI-TOD-v2 Full Empirical Benchmark Matrix (12-GPU Kaggle Cluster)

**Date**: 2026-08-24  
**Status**: 🚀 **ACTIVELY RUNNING** across 12 dedicated Kaggle Tesla T4 GPU Accounts  
**Protocol**: AI-TOD-v2 Official 1x Schedule (12 Epochs, SGD $\eta=0.005$, Batch Size 2, Cosine Decay)  
**Dataset**: `simplestzyp/tiny-object-detection-in-aerial-images` (28,036 images, 700,621 annotations across 8 categories)

---

## 1. Complete 12-GPU Cluster Dispatch Matrix

| Experiment ID | Method | Mechanism | Kaggle Account | Kernel Reference Slug | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `aitod_baseline_s42` | **Faster R-CNN** (ICCV'15) | Standard IoU / Smooth-L1 | `amongus1504` | `amongus1504/tod-aitod-baseline-s42-20260823` | 🏃 `RUNNING` |
| `aitod_nwd_s42` | **NWD** (NeurIPS'21) | 2-Wasserstein Gaussian Distance | `dipphmngc` | `dipphmngc/tod-aitod-nwd-s42-20260823` | 🏃 `RUNNING` |
| `aitod_igwd_s42` | **IGWD** (IEEE TMM'22) | Scale-Invariant Gaussian Metric | `hienquang06` | `hienquang06/tod-aitod-igwd-s42-20260823` | 🏃 `RUNNING` |
| `aitod_rfla_s42` | **RFLA** (ECCV'22) | Receptive Field Gaussian Assign | `hngngnguynvn` | `hngngnguynvn/tod-aitod-rfla-s42-20260823` | 🏃 `RUNNING` |
| `aitod_hwiou_sig8_s42` | **H-WIoU** ($\sigma_0=8.0\text{px}$) | Proposed Scale Homotopy Manifold | `quangnhtng` | `quangnhtng/tod-aitod-hwiou-sig8-s42-20260823` | 🏃 `RUNNING` |
| `aitod_hwiou_sig6_s42` | **H-WIoU** ($\sigma_0=6.0\text{px}$) | Proposed Scale Homotopy Manifold | `qnhat1504` | `qnhat1504/tod-aitod-hwiou-sig6-s42-20260823` | 🏃 `RUNNING` |
| `aitod_hwiou_sig10_s42` | **H-WIoU** ($\sigma_0=10.0\text{px}$) | Proposed Scale Homotopy Manifold | `thyngluthy` | `thyngluthy/tod-aitod-hwiou-sig10-s42-20260823` | 🏃 `RUNNING` |
| `aitod_cascade_s42` | **Cascade R-CNN** (CVPR'18) | Multi-Stage Bounding Box Cascade | `hngtrngtn` | `hngtrngtn/tod-aitod-cascade-s42-20260824` | 🏃 `RUNNING` |
| `aitod_dotd_s42` | **DotD** (ICCV'21) | Normalized Dot-Distance Regression | `luongsythanh` | `luongsythanh/tod-aitod-dotd-s42-20260824` | ⏳ `QUEUED -> RUNNING` |
| `aitod_simd_s42` | **SimD** (CVPR'23) | Similarity Distribution Assigner | `pptlyn11` | `pptlyn11/tod-aitod-simd-s42-20260824` | ⏳ `QUEUED -> RUNNING` |
| `aitod_safit_s42` | **SAFit** (AAAI'24) | Scale-Adaptive Feature Integration | `trieuvo123` | `trieuvo123/tod-aitod-safit-s42-20260824` | ⏳ `QUEUED -> RUNNING` |
| `aitod_hwiou_cascade_s42` | **H-WIoU + Cascade** (Ours) | Multi-Stage Homotopy Hybrid | `phuc1806` | `phuc1806/tod-aitod-hwiou-cascade-s42-20260824` | ⏳ `QUEUED -> RUNNING` |

---

## 2. Evaluation Protocol & Official Benchmark Metrics

Every model will evaluate each epoch using the official `aitodpycocotools` evaluator computing:
1. **Mean Average Precision**: $\text{AP}$, $\text{AP}_{50}$, $\text{AP}_{75}$
2. **Scale Partition Metrics**:
   - $\text{AP}_{vt}$ (Very Tiny: $2\text{--}8\text{ px}$)
   - $\text{AP}_{t}$ (Tiny: $8\text{--}16\text{ px}$)
   - $\text{AP}_{s}$ (Small: $16\text{--}32\text{ px}$)
   - $\text{AP}_{m}$ (Medium: $32\text{--}64\text{ px}$)
3. **Average Recall**: $\text{AR}_{100}$, $\text{AR}_{1500}$
4. **Per-Category $\text{AP}_{50}$ Breakdown**: Airplane, Bridge, Storage-tank, Ship, Swimming-pool, Vehicle, Person, Wind-mill.

---

## 3. Automation Tooling

- **Dispatcher**: `paper_a/tools/launch_remaining_5_accounts.py`
- **Real-Time Monitor & Downloader**: `paper_a/tools/continuous_cluster_monitor.py`
- **Artifact Destination**: `journal/results/aitod_empirical/`

