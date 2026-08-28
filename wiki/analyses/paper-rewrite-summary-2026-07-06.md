---
title: Paper Rewrite Summary - 2026-07-06
type: analysis
created: 2026-07-06
updated: 2026-07-06
sources: [paper/main.tex, paper/saalw_main.tex, runs/test_results.json]
tags: [paper, alw, sa-alw, rewrite, coco-metrics]
---

## Paper Rewrite Summary — 2026-07-06

### Background
The original ALW paper draft used conference IEEEtran format, old notebook numbers (mAP=0.5439), custom scale-aware metrics, and contained SA-ALW/cascade content mixed in. The SA-ALW paper had not been formally written.

### What Was Done

#### 1. ALW Paper — Complete Restructure
- **Class**: conference → journal IEEEtran for TIP submission
- **Metrics**: switched from custom scale-aware (mAP(scale), AP_micro/tiny/small/large) to standard COCO (AP, AP50, AP75, AP_S, AP_M, AP_L, AR100)
- **Tables removed**: validation tables, training trajectory tables, computational cost tables, old hyperparameter tables
- **Single test table**: 7 COCO metrics × all configurations
- **Component ablation table**: IGWD→ALW decomposition (anisotropy vs log-shape) on test set
- **Honest reporting**: ALW wrapped overfits (val 0.5835→test 0.4572, Δ=-0.126), noted as footnote
- **Property table**: IoU, NWD, IGWD, GCD, ALW — 4 properties (position sensitivity, scale invariance, anisotropy, shape space)

#### 2. SA-ALW Paper — New from Scratch
- **Method**: ALW derivation from first principles + SA-ALW extension (β(s), w_pos(s))
- **Three theorems**: symmetry, dimensional consistency, scale invariance
- **Property table**: extended to 5 columns (adds "Adaptive" column)
- **Component ablation**: 3 levels — IGWD→ALW, ALW→SA-ALW β-only, ALW→SA-ALW w_pos-only, SA-ALW full
- **Cascade removed**: pipeline not yet breakthrough, reserved for future paper
- **All numbers verified**: against runs/test_results.json and runs/*/metrics.csv

#### 3. LaTeX Diagnostics
- `\tnote` → `$^{a}$` (removes undefined command)
- `\texorpdfstring` on captions with math
- `\hbadness`/`\vbadness` thresholds added
- Property tables: `\scriptsize` + `\setlength{\tabcolsep}`
- `\,px` consistency across both papers
- Removed unused bib entries (yolov8, solovyev2021wbf)
- Fixed SA-ALW limitations debris (copy-paste from cascade doc removed)

#### 4. Wiki Updated
- Wiki overview updated with final COCO results tables
- Phase status changed from "in progress" to "completed"
- Next steps: pure ALW train, multi-seed, cross-dataset, cascade breakthrough

### Files Changed
| File | Change |
|------|--------|
| `paper/main.tex` | Full rewrite — TIP journal, COCO metrics, updated text |
| `paper/experiments.tex` | Strip val/trajectory tables, single COCO test table |
| `paper/saalw_main.tex` | New — SA-ALW paper, ALW+SA-ALW method, 3 theorems |
| `paper/saalw_experiments.tex` | New — COCO test table, ablation tables |
| `paper/references.bib` | Remove unused entries, clean up |
| `wiki/overview.md` | Updated to current state |
| `wiki/analyses/paper-rewrite-summary-2026-07-06.md` | This file |

### Numbers Verified
All ~430 numeric values in both papers cross-checked against:
- `runs/test_results.json` — all COCO metrics
- `runs/*/metrics.csv` — per-epoch validation
- `eda/Phase0_report.md` — dataset statistics
- `wiki/analyses/Phase 2-4 Results Summary - 2026-07-04.md` — phase results

### Open Items
1. Pure ALW training run (no Charbonnier wrapper) for clean test evaluation
2. Multi-seed statistics (123, 2024) for ALW, SA-ALW, IGWD
3. Cross-dataset validation (AI-TOD, AI-TOD-v2)
4. IGWD reference bibliographic verification
5. LaTeX compile: IEEEtran.cls needs MiKTeX install (offline environment)
6. No figures yet — method diagram, qualitative examples needed before submission
