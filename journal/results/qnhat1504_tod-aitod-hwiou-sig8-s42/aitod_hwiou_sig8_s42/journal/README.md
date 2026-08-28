# Homotopy Wasserstein-IoU (H-WIoU) - Dedicated Journal Workspace

This directory contains the independent, self-contained journal manuscript, publication assets, experimental tools, and statistical verification ledgers targeting **IEEE TPAMI / IJCV / IEEE TMM**.

---

## Directory Structure

```
journal/
├── manuscript/                  # LaTeX manuscript package
│   ├── main.tex                 # Primary IEEE Transactions / Nature style source
│   ├── main.pdf                 # Compiled 5-page publication-grade PDF
│   └── figures/                 # Embedded vector PDFs and 300 DPI figures
│       ├── fig1_homotopy_theory.pdf
│       ├── fig2_multimetric_radar.pdf
│       └── fig3_ablation_landscape.pdf
├── figures/                     # Standalone high-res figure assets
├── results/                     # Empirical statistical ledgers and JSON audits
│   └── statistical_significance_audit.json
└── tools/                       # Reproducibility tools
    ├── build_figures.py         # Pure Matplotlib 300 DPI & vector PDF generator
    ├── compute_statistics.py    # Pure NumPy Paired t-test, Wilcoxon, Bootstrap CIs
    └── verify_results.py        # Kaggle log streaming parser and multi-metric validator
```

---

## Build & Verification Commands

To regenerate figures, compute statistics, and recompile the manuscript PDF:

```powershell
# 1. Regenerate publication figures
.\.venv-cuda\Scripts\python.exe journal\tools\build_figures.py

# 2. Compute statistical hypothesis tests & bootstrap CIs
.\.venv-cuda\Scripts\python.exe journal\tools\compute_statistics.py

# 3. Recompile LaTeX manuscript PDF
cd journal\manuscript
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..\..
```
