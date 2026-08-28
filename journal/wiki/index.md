---
title: "Journal Project Wiki: Homotopy Wasserstein-IoU (H-WIoU)"
type: "index"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
  - "common/metrics/iou.py"
  - "journal/tools/render_paperbanana_hwiou_diagram.py"
tags:
  - "journal"
  - "h-wiou"
  - "tiny-object-detection"
  - "index"
---

# Journal Project Wiki: Homotopy Wasserstein-IoU (H-WIoU)

Welcome to the dedicated **Journal Research Wiki** for **Homotopy Wasserstein-IoU (H-WIoU)** — a scale-invariant detection framework for tiny and microscopic object detection targeted for top-tier computer vision publication (IEEE TPAMI / IJCV format).

This wiki serves as the isolated, authoritative knowledge hub for all mathematical derivations, empirical validation ledgers, architectural blueprints, and experimental benchmarks of the journal project.

---

## 🗺️ Knowledge Map & Table of Contents

### 1. Core Overview & Activity Diary
* [[overview]]: Executive summary, problem definition, core formulation, and primary empirical takeaways.
* [[log]]: Chronological experiment, derivation, cluster deployment, and manuscript revision diary.

### 2. Theoretical Concepts & Mathematical Proofs
* [[concepts/homotopy_wasserstein_iou]]: Topological $C^\infty$ homotopy formulation bridging discrete Lebesgue measures and Riemannian optimal transport.
* [[concepts/gradient_asymptotics_and_boundedness]]: Non-vanishing gradient bound proofs ($\|\nabla_\theta \mathcal{L}_{\text{H-WIoU}}\| = \mathcal{O}(1)$ under disjoint alignment vs $\nabla \mathcal{L}_{\text{IoU}} \equiv 0$).
* [[concepts/homotopy_label_assignment]]: Stage 1 RPN dynamic label assignment mechanism and candidate positive survival rate analysis ($0.18 \to 0.94$).
* [[concepts/bounded_homotopy_box_loss]]: Stage 2 RoI Head bounded regression loss $\mathcal{L}_{\text{H-WIoU}} = 1 - \mathcal{S}_{\text{H-WIoU}}$ with scale-adaptive boundary fitting.

### 3. Empirical Analyses & Benchmark Results
* [[analyses/tinyperson_empirical_benchmark]]: Fair-20 protocol evaluation on maritime TinyPerson benchmark ($+5.91\%\ \text{mAP}_{50}$, $+3.09\%\ \text{AP}_{\text{micro}}$).
* [[analyses/aitodv2_sota_matrix]]: Official 8-class AI-TOD-v2 aerial benchmark comparison ($\text{AP}_{vt} = 5.72\%$, $\text{AR}_{vt} = 11.27\%$).
* [[analyses/hwiou_v2_roadmap_and_extensions]]: Comprehensive mathematical foundations and roadmap for 3 novel extensions (DU-HWIoU, SW-HWIoU, O-HWIoU).
* [[analyses/ablation_studies_and_sensitivity]]: 4-axis ablation on scale threshold $\sigma_0$, transition geometries, and module placement isolation.
* [[analyses/statistical_significance_and_bootstrap]]: Paired Student's $t$-test ($t = 73.18, p < 10^{-20}$), Wilcoxon test, and $N=10,000$ non-parametric bootstrap confidence intervals.

### 4. Syntheses & Publication Artifacts
* [[syntheses/journal_manuscript_architecture]]: 8-page IEEE double-column manuscript breakdown, table schemas, and LaTeX compilation pipeline.
* [[syntheses/visual_diagram_paperbanana_design]]: PaperBanana-standard visual-first architecture diagram design (Figure 5), 3D tensor cuboids, and zoom loupes.

---

## 📐 Quick Reference: Key Formulations

$$\mathcal{S}_{\text{H-WIoU}}(A, B) = [\text{IoU}(A, B)]^{\gamma(s_B)} \cdot \exp\left(-(1 - \gamma(s_B))\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$$

where the continuous scale homotopy parameter is given by:

$$\gamma(s) = \frac{s^2}{s^2 + \sigma_0^2} \in [0, 1), \quad s = \sqrt{w \cdot h}$$

* $\lim_{s \to 0^+} \gamma(s) = 0 \implies \mathcal{S}_{\text{H-WIoU}} \to \exp(-\mathcal{D}_{\mathcal{W}}^2)$ (Pure Optimal Transport)
* $\lim_{s \to \infty} \gamma(s) = 1 \implies \mathcal{S}_{\text{H-WIoU}} \to \text{IoU}$ (Pure Discrete Lebesgue Measure)
