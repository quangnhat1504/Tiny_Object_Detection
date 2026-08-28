# Implementation Plan: Conference Paper Completion & Publication Diagrams

## Overview
Complete the full manuscript draft for the conference submission, integrating the completed 21-model 20-epoch empirical benchmark, formalizing the mathematical framework (SA-ALW, Iterative-CBL, PC-MR, PC-MOC, and Joint Model), and generating publication-grade vector/high-resolution diagrams and performance figures.

## Architecture Decisions
- **Manuscript Scope**: Paper A (`paper_a/manuscript/`) is updated to be fully complete, self-contained, and mathematically rigorous.
- **Figures & Visualizations**:
  - `fig1_framework_architecture.png` / `.pdf`: High-definition schematic of the complete Joint Detection Framework (Micro-Rescue RPN Gradient Projection + Multilevel FPN Distillation + Dynamic Scale Routing).
  - `fig2_geometry_comparison.png` / `.pdf`: Geometric illustration comparing Standard IoU, NWD Gaussian approximation, and SA-ALW Anisotropic Log-Wasserstein metric on sub-8px objects.
  - `fig3_megabenchmark_comparison.png` / `.pdf`: Multi-metric bar chart with $\pm \text{std}$ error bars across all 7 methods $\times$ 3 seeds.
  - `fig4_convergence_routing.png` / `.pdf`: Training convergence trajectories across 20 epochs for all methods.
- **Statistical Rigor**: Embed exact 21-model results with paired $t$-tests and 10,000-resample Bootstrap 95% Confidence Intervals.

## Task List & Dependency Graph

### Phase 1: High-Resolution Publication Figure Generation
- [ ] Task 1: Generate Framework Architecture Diagram (`fig1_framework_architecture`)
- [ ] Task 2: Generate Metric Geometry & Gradient Behavior Diagram (`fig2_geometry_comparison`)
- [ ] Task 3: Generate 21-Model Statistical Benchmark Chart with Error Bars (`fig3_megabenchmark_comparison`)
- [ ] Task 4: Generate Training Convergence & Scale-Loss Trajectory Plot (`fig4_convergence_routing`)

### Checkpoint: Figures
- [ ] All 4 figures generated as high-resolution PNG (300 DPI) and PDF vector formats in `paper_a/figures/` and artifact directory.

### Phase 2: Manuscript Text & Mathematical Formalization
- [ ] Task 5: Rewrite Abstract & Introduction (`sections/introduction.tex`, `main.tex`)
- [ ] Task 6: Modernize Related Work (`sections/related_work.tex`)
- [ ] Task 7: Complete Mathematical Formulations in Method Section (`sections/method.tex`)
  - Explicit equations for SA-ALW distance & temperature decay
  - Formulation of Iterative-CBL dynamic routing
  - Formulation of PC-MR orthogonal gradient projection ($\nabla \mathcal{L}_{\text{proj}}$)
  - Formulation of PC-MOC feature distillation & cosine loss
- [ ] Task 8: Update Experiments & Statistical Mega-Table in Experiments Section (`sections/experiments.tex`)
  - Integration of Master Mega-Table (Table 1)
  - Statistical significance & Paired t-test analysis
  - Component ablation study
- [ ] Task 9: Update Discussion, Limitations & Future Work (`sections/limitations.tex`)

### Checkpoint: Manuscript Verification
- [ ] Review entire LaTeX document structure, verify all table references and figure inclusions.

## Risks and Mitigations
| Risk | Impact | Mitigation |
|---|---|---|
| LaTeX compilation dependencies missing | Medium | Ensure portable standard LaTeX packages (`graphicx`, `amsmath`, `booktabs`, `subcaption`) without proprietary fonts. |
| Figure scaling issues in two-column format | Low | Design all figures specifically for single-column width ($0.48\textwidth$) or double-column span ($0.98\textwidth$). |
