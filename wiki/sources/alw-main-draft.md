---
title: ALW Main Draft
type: source
created: 2026-06-10
updated: 2026-06-10
sources: [raw/main.pdf, raw/main.extracted.txt, tod-alw.ipynb]
tags: [alw, paper-draft, tiny-object-detection, manuscript]
---

## ALW Main Draft

`raw/main.pdf` is the current 5-page ALW manuscript draft titled "ALW: An Anisotropic Log-Wasserstein Distance for Tiny Object Detection". The draft positions ALW as a parameter-free bounding-box distance for tiny object detection, replacing prior Gaussian metrics' Euclidean shape terms and isotropic center normalization with:

- per-axis RMS normalizers: `Sx = (wp^2 + wt^2) / 2`, `Sy = (hp^2 + ht^2) / 2`
- squared log-ratio shape terms: `log(wp/wt)^2`, `log(hp/ht)^2`

The paper uses ALW in label assignment and box regression, with standard IoU-NMS held fixed for the reported results.

## Current Paper Claim

The current claim is strongest as:

> ALW improves strict localization and recall over IGWD under an otherwise identical Faster R-CNN + RFLA training harness on SOD/TinyPerson-Sea, while remaining competitive with GCD.

Current key reported numbers:

| Method | mAP(scale) | mAP@50 | AP_micro | AP_tiny | AP_large | COCO AP50:75 | COCO AP75 | AR@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IGWD | 0.5187 | 0.3294 | 0.1928 | 0.5084 | 0.7075 | 0.1639 | 0.0312 | 0.3650 |
| GCD | 0.5522 | 0.3483 | 0.2582 | 0.5437 | 0.7455 | 0.1821 | 0.0415 | 0.3939 |
| ALW | 0.5439 | 0.3435 | 0.2495 | 0.5283 | 0.7956 | 0.1893 | 0.0536 | 0.4003 |

The cleanest comparison is IGWD -> ALW because the draft says that harness is controlled. The NWD/GCD/IGWD table is currently marked as local-table results and still needs re-validation.

## What Is Already In The Draft

- Abstract, introduction, related work, method, experiments, limitations, conclusion, reproducibility note, and references.
- Mathematical motivation for log-ratio shape and anisotropic per-axis position normalization.
- Properties: symmetry, dimensional consistency, scale invariance.
- Algorithm sketch for ALW hierarchical label assignment.
- SOD/TinyPerson-Sea dataset statistics.
- Main results table and ALW validation trajectory.

## Important Weaknesses In The Draft

- The paper calls ALW a "metric", but currently proves symmetry, dimensional consistency, and scale invariance, not triangle inequality. Either prove metric properties formally or use "distance/similarity" more carefully.
- Table I and the text may conflict around IGWD scale invariance. Table I marks IGWD as scale-invariant, while the text says IGWD only approximately cancels Euclidean shape scale effects. This needs a precise correction.
- NWD/GCD/IGWD baselines are not yet all byte-identical re-runs.
- Only one seed is reported.
- ALW's two claimed components are not separately ablated.
- Public benchmark validation on AI-TOD / AI-TOD-v2 / VisDrone is not yet present.
- Metric-NMS and beta sweeps are explicitly future work.
- The bibliography still contains placeholder/uncertain entries, especially IGWD.

## Ingest Notes

Text was extracted to `raw/main.extracted.txt` using `pypdf`. The extracted PDF text has minor spacing/encoding artifacts, but the scientific content is readable.
