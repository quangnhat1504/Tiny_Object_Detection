# ALW / SA-ALW Legacy Drafts

> **Diagnostic history only.** The conference-refinement source of truth is
> `../paper_a/`. Numerical claims and tables in this directory use legacy
> tile-level/reused-test protocols and must not be copied into a submission.

Historical drafts for ALW and SA-ALW.

## Files
- `main.tex` — full paper (IEEEtran conference format).
- `references.bib` — bibliography.
- `saalw_main.tex` / `saalw_experiments.tex` — SA-ALW paper draft.
- `checkpoints/iterative_cbl_fair20_2026-08-01.md` — frozen evidence package
  for the next SA-ALW + trainable iterative-CBL paper. It is not yet merged
  into either existing LaTeX draft.
- `checkpoints/performance_research_2026-08-02.md` — current paper handoff for
  PC-MR-RPN, PC-MOC-FD, the rejected PC-MHFD branch, the compatibility-only
  combination, and the still-running fair-20 matrix.

## Build
No LaTeX toolchain is installed in this environment, so the PDF was not compiled here. To build (TeX Live / MiKTeX with `IEEEtran`, `pifont`, `algorithm`, `algpseudocode`, `booktabs`):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
or `latexmk -pdf main.tex`. To target a CVPR/ICCV submission later, swap the document class for `cvpr.sty`/`iccv.sty`; the body is class-agnostic.

## Scope (important)
The scope below describes the historical draft, not the current Paper A
contract. See `../paper_a/scope_contract.md` for the frozen scope.
This draft covers **only the published baseline + ALW story**, per project instructions:
- **Proposed method:** ALW (anisotropic per-axis position normalization + log-ratio shape), used in **RFLA label assignment** and **RoI box-regression loss**.
- **Baselines:** NWD, IGWD, GCD (and IoU for context), RFLA as the assignment skeleton.
- **Deliberately excluded** (reserved for the next paper): SAC backbone, HFP/SDP FPN, P2/stride-4 features, SAH-GD hybrid metrics, dynamic top-k / reliability-gated robust shape, dual-objective regression. The actual `tod-alw.ipynb` run uses a plain ResNet-50-FPN, so the paper describes exactly that.

## Where the numbers come from
- **ALW results** (mAP(scale)=0.5439 @ epoch 8, COCO AP75=0.0536, etc.) are read directly from the executed output of `../tod-alw.ipynb` on the maritime TinyPerson-Sea dataset (`sod-tinypeopleinsea`, classes dry-person / wet-swimmer).
- **NWD / GCD / IGWD** numbers are from the project local metric-comparison table (`wiki/sources/tiny-object-metrics-comparison-filled.md`) under the same evaluation protocol.
- **Dataset stats** are from `eda/REPORT.md`.

## Honesty / to-verify before submission (see Sec. "Limitations")
- The **IGWD → ALW** comparison is byte-identical (only the metric function changes) — this is the clean, fully controlled result. NWD/GCD rows should be re-run on the identical harness for camera-ready.
- Single seed, single dataset. Add seeds (mean±std) and AI-TOD / AI-TOD-v2 validation.
- Per-component ablation (anisotropy vs. log-shape, in isolation) is **not yet run** — described as planned.
- `references.bib` `igwd2024` entry is a placeholder from the source PDF; confirm venue/authors.
- NMS uses standard IoU-NMS in all runs (held fixed); ALW-NMS is future work, not claimed as a result.

## Headline framing
ALW is the only metric in the Gaussian family that is simultaneously position-sensitive, exactly scale-invariant, anisotropic, and log-space in shape. Empirically it **beats its direct predecessor IGWD on every metric** (AP75 +72% relative) and leads all baselines on **strict localization (AP75), COCO AP50:75, recall (AR@100), and large-object AP**, while staying within 0.8 pt of the strongest baseline (GCD) on coarse mAP@50.
