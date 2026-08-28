# Task List: Paper Draft Completion & Scientific Diagrams

## Phase 1: High-Resolution Publication Figure Generation
- [x] **Task 1: Generate Framework Architecture Diagram (`fig1_framework_architecture`)**
  - **Description**: Create a publication-ready schematic diagram detailing the Faster R-CNN backbone with RPN Proposal Micro-Rescue (PC-MR), Orthogonal Gradient Projection, and Multilevel FPN Feature Distillation (PC-MOC).
  - **Acceptance Criteria**:
    - High-res PNG (300 DPI) and PDF saved to `paper_a/figures/`
    - Clear visual distinction between backbone, RPN, gradient projection, and RoI head
  - **Verification**: `paper_a/tools/generate_publication_figures.py` generated `fig1_framework_architecture.png` and `.pdf`.

- [x] **Task 2: Generate Metric Geometry & Gradient Behavior Diagram (`fig2_geometry_comparison`)**
  - **Description**: Visualize the 2D bounding box IoU collapse vs. NWD Gaussian blurring vs. SA-ALW Anisotropic Log-Wasserstein distance landscape under pixel offsets from 0 to 10px on $6\times 6$ px bounding boxes.
  - **Acceptance Criteria**:
    - Shows loss smoothness and gradient magnitude as a function of translation distance.
    - Demonstrates why standard IoU has zero gradient at non-overlapping offsets while SA-ALW provides smooth continuous feedback.
  - **Verification**: `paper_a/tools/generate_publication_figures.py` generated `fig2_geometry_comparison.png` and `.pdf`.

- [x] **Task 3: Generate 21-Model Statistical Benchmark Chart with Error Bars (`fig3_megabenchmark_comparison`)**
  - **Description**: Multi-panel publication bar chart comparing all 7 methods across $mAP_{50}$, $mAP_{primary}$, $coco\_AP_{75}$, $AP_{micro}$, $AP_{tiny}$ with error bars denoting $\pm 1 \text{ std}$ across 3 independent random seeds.
  - **Acceptance Criteria**:
    - Color-coded by category (External Baseline, SOTA Metric, Predecessor, Proposed Components, Proposed Full Model).
    - Annotates $+5.07\%$ gain on $AP_{micro}$ and $+1.40\%$ on $coco\_AP_{75}$.
  - **Verification**: `fig3_megabenchmark_comparison.png` and `.pdf` generated and verified.

- [x] **Task 4: Generate Training Convergence & Scale-Loss Trajectory Plot (`fig4_convergence_routing`)**
  - **Description**: Plot loss convergence curves and mAP progress over 20 epochs for Standard Faster R-CNN, NWD, Iterative-CBL, and the Joint Model.
  - **Acceptance Criteria**:
    - Parsed directly from actual training `metrics.csv` files across runs.
  - **Verification**: `fig4_convergence_trajectories.png` and `.pdf` generated.

## Checkpoint: Figures
- [x] All 4 figures generated in `paper_a/figures/` and copied to artifact directory.

## Phase 2: Manuscript Text & Mathematical Formalization
- [x] **Task 5: Rewrite Abstract & Introduction (`sections/introduction.tex`, `main.tex`)**
  - **Description**: Update abstract and introduction with precise problem statements (spatial discretization noise, scale imbalance, micro-instance gradient annihilation) and highlight our 21-model experimental verification.
  - **Acceptance Criteria**: Fully articulated motivation, 3 major contributions, and crisp roadmap.

- [x] **Task 6: Modernize Related Work (`sections/related_work.tex`)**
  - **Description**: Structure related work into Tiny Object Detection, Wasserstein & Metric-Based Losses, and Gradient/Feature Distillation in Multi-Scale Networks.

- [x] **Task 7: Complete Mathematical Formulations in Method Section (`sections/method.tex`)**
  - **Description**: Detail the full mathematical derivations of SA-ALW, Iterative-CBL dynamic routing, PC-MR gradient projection ($\mathbf{g}_{\text{proj}} = \mathbf{g}_{\text{micro}} - \frac{\mathbf{g}_{\text{micro}} \cdot \mathbf{g}_{\text{main}}}{\|\mathbf{g}_{\text{main}}\|^2} \mathbf{g}_{\text{main}}$), and PC-MOC cosine feature distillation loss.

- [x] **Task 8: Update Experiments & Statistical Mega-Table in Experiments Section (`sections/experiments.tex`)**
  - **Description**: Embed 21-model master mega-table, paired $t$-test values, bootstrap confidence intervals, and component ablation analysis.

- [x] **Task 9: Update Discussion, Limitations & Future Work (`sections/limitations.tex`)**
  - **Description**: Discuss computational overhead, anchor density trade-offs, and future directions.

## Checkpoint: Complete
- [x] Full manuscript compiled with `pdflatex` to 8-page publication PDF (`Paper_A_Manuscript_MegaBenchmark.pdf`).
