# Homotopy Wasserstein-IoU (H-WIoU) - Dedicated Journal Workspace

This directory contains the independent, self-contained journal manuscript, publication assets, experimental tools, and statistical verification ledgers targeting **IEEE TPAMI / IJCV / IEEE TMM**.

### Authors
1. **Đặng Quang Nhật** (DE200497) - `dangquangnhat1504@gmail.com` - 0377231436 (First & Corresponding Author)
2. **Lê Hồ Anh Duy** (DE200171) - `lehoanhduy5426@gmail.com` - 0898896962
3. **Phạm Minh Tiến** (DE191091) - `taxaceae.forwork@gmail.com` - 0968338702
*Affiliation: Department of Artificial Intelligence & Computer Science, FPT University*

---

## Directory Structure

```
journal/
├── wiki/                        # Dedicated Journal Research Wiki & Knowledge Base
│   ├── index.md                 # Master Table of Contents & Ontology Map
│   ├── overview.md              # Executive Summary & Theoretical Core
│   ├── log.md                   # Chronological Research & Experiment Diary
│   ├── concepts/                # Mathematical proofs, HLA, Bounded loss
│   ├── analyses/                # TinyPerson, AI-TOD-v2, Ablations, Bootstrap
│   └── syntheses/               # Manuscript blueprint & PaperBanana design
├── manuscript/                  # LaTeX manuscript package (8-page IEEE TPAMI)
│   ├── main.tex                 # Primary IEEE Transactions style source
│   ├── main.pdf                 # Compiled publication-grade PDF
│   └── figures/                 # Embedded vector PDFs and 300 DPI figures
├── figures/                     # Standalone high-res figure assets (Figs 1-5)
├── results/                     # Empirical statistical ledgers and JSON audits
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
