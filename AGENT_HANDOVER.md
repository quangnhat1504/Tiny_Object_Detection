# AGENT HANDOVER BRIEFING: TINY OBJECT DETECTION (EH-WIoU)

> **Directive for Incoming Coding Agent**: Read this document first. All decisions, mathematical formulations, and engineering architecture below reflect the current verified state of the codebase. Do not reinvent past experiments or fragment compute across multiple seeds. Follow the **Seed 42 Single-Seed Protocol**.

---

## 1. Project Context & Objectives
- **Target Venue**: IEEE TPAMI (Transactions on Pattern Analysis and Machine Intelligence) / High-impact Computer Vision publication.
- **Problem**: Detecting tiny and microscopic objects ($s < 8\text{px}$) in high-resolution aerial (AI-TOD-v2) and maritime (TinyPerson) imagery.
- **Core Challenge**: Discrete $\text{IoU}$ drops to zero when bounding boxes are slightly displaced, causing gradient collapse and severe anchor starvation. Optimal transport (NWD) blurs boundaries.
- **Our Solution**: **Entropy-Modulated Homotopy Wasserstein-IoU (EH-WIoU)** — a smooth convex homotopy between 2-Wasserstein optimal transport and discrete IoU.

---

## 2. Strict Ground Rule: Seed 42 Single-Seed Protocol
- **User Instruction**: *"mình nghĩ chúng ta nên cứ tập trung 1 seed trước khi đi sang seed khác đi nhé, tiếp tục audit code, tìm keypoint, phát triển lên"*.
- **Rule**: Prioritize **Seed 42** exclusively for all ablations, pipeline upgrades, and architectural validation. Do NOT launch multi-seed sweeps (e.g. 123, 2024) until the Seed 42 configuration reaches maximum empirical optimality.

---

## 3. Mathematical Foundations

### 3.1 Additive Convex Homotopy
$$\mathcal{S}_{\text{EH-WIoU}}(\mathbf{b}_a, \mathbf{b}_g) = \gamma(s_g, H) \cdot \text{IoU}(\mathbf{b}_a, \mathbf{b}_g) + (1 - \gamma(s_g, H)) \cdot \mathcal{S}_W(\mathbf{b}_a, \mathbf{b}_g)$$

- **Characteristic Scale Parameter**:
  $$\gamma(s_g, H) = \frac{s_g^2 (1 + \beta H)}{s_g^2 (1 + \beta H) + \sigma_0^2}, \quad s_g = \sqrt{w_g \cdot h_g}, \quad \sigma_0 = 8.0\text{px}$$
- **Micro Regime ($s_g \to 0$)**: $\gamma \to 0 \implies \mathcal{S} \to \mathcal{S}_W$ (Non-vanishing gradient $\|\nabla \mathcal{L}\| = \mathcal{O}(1)$).
- **Macro Regime ($s_g \to \infty$)**: $\gamma \to 1 \implies \mathcal{S} \to \text{IoU}$ (Strict boundary tightness).

### 3.2 Euclidean Gaussian 2-Wasserstein (No Logarithmic Divergence)
Bounding boxes modeled as $\mathcal{N}(\mathbf{c}, \Sigma)$ with $\Sigma = \operatorname{diag}(w^2/4, h^2/4)$:
$$\mathcal{W}_2^2 = \|\mathbf{c}_a - \mathbf{c}_g\|_2^2 + \frac{1}{4}\left((w_a - w_g)^2 + (h_a - h_g)^2\right)$$
$$\mathcal{D}_W^2 = \frac{\mathcal{W}_2^2}{2 s_g^2 + \epsilon}, \quad \mathcal{S}_W = \exp\left(-\mathcal{D}_W^2\right) \in (0, 1]$$

---

## 4. Key Diagnostic Insights & Recent Upgrades

### Empirical Test Baseline (14,018 official test images on RTX 5070 Ti)
- EH-WIoU ($\sigma_0=8.0$, Seed 42): **$\text{mAP} = 21.68\%$**, $\text{mAP}_{50} = 44.59\%$, $\text{mAP}_{75} = 20.72\%$, $\text{AP}_{vt} = 7.00\%$, $\text{oLRP} = 0.8289$.
- **Bottleneck Identified**: **$\text{oLRP}_{\text{fn}} = 0.6172$** $\implies$ **61.7% of all errors are False Negatives (missed objects)**.

### Upgrades Implemented & Certified
1. **Homotopy-Aware RoI Head Matching** (`common/model.py: _wrap_roi_for_homotopy_matching`):
   - *Why*: Stage 1 RPN generates high-quality micro-proposals, but torchvision Stage 2 RoIHead used discrete $\text{IoU} \ge 0.50$. A $6\times 6\text{px}$ box shifted by $2\text{px}$ has $\text{IoU} = 0.2857 < 0.50$ and was assigned to BACKGROUND (`0`), cutting off regression loss.
   - *Fix*: Scale-conditioned quality blend $\mathcal{Q} = (1 - \alpha)\mathcal{S}_{\text{EH-WIoU}} + \alpha \text{IoU}$ with `Matcher(0.40, 0.30, allow_low_quality_matches=True)`. Micro-proposals now survive as positive RoI targets.
2. **Feature-Level Entropy Guidance Module (EGM)** (`common/model.py: FPNEntropyGuidance`):
   - Attached to FPN $P_2$ (stride 4) and $P_3$ (stride 8) to amplify spatial detail and high-frequency contrast. Registered as PyTorch submodule `base.backbone.egm`.
3. **Cascade Homotopy Harmonization** (`common/metrics/cascade_homotopy.py`):
   - Unified with Euclidean 2-Wasserstein formulation and multi-stage refinement ($\sigma = [8.0, 4.0, 2.0]$).
4. **Checkpoint Bugfix** (`scripts/train_frcnn_aitod.py`):
   - Fixed defect where early zero-detection epochs unconditionally overwrote `best.pt`.
   - Exposed `--use-homotopy-roi` and `--use-egm` CLI options.

---

## 5. Repository File Map

| Path | Purpose & Responsibilities |
| :--- | :--- |
| `common/metrics/entropy_homotopy.py` | Core EH-WIoU metric, memory-safe chunking ($N \le 16384$), bounded loss. |
| `common/metrics/cascade_homotopy.py` | Multi-stage cascade homotopy loss and stage assigner. |
| `common/model.py` | Faster R-CNN builder (`build_model`), `FPNEntropyGuidance`, RoI wrappers. |
| `scripts/train_frcnn_aitod.py` | Official AI-TOD-v2 training and evaluation script. |
| `paper_a/tests/test_homotopy_roi_matching.py` | Unit tests for RoI matching survival and EGM backpropagation. |
| `journal/results/ehwiou_downloaded_evaluated_metrics.json` | Authoritative 14,018-test-image evaluation results. |
| `journal/wiki/` | Research diary, theoretical concept derivations, and empirical logs. |

---

## 6. Verification Commands (Always Run Before Any Push)

```powershell
# 1. Run all 107 unit tests (must pass 107/107)
.\.venv-cuda\Scripts\python.exe -m unittest discover -s paper_a\tests -p "test_*.py"

# 2. Syntax check core files
.\.venv-cuda\Scripts\python.exe -m py_compile common\model.py common\metrics\cascade_homotopy.py scripts\train_frcnn_aitod.py

# 3. Validate Phase 0 ledger contracts
.\.venv-cuda\Scripts\python.exe paper_a\tools\validate_phase0.py
```

---

## 7. How to Launch Upgraded Training (Seed 42)

```powershell
.\.venv-cuda\Scripts\python.exe scripts\train_frcnn_aitod.py `
    --metric eh_wiou `
    --placement h_wiou `
    --box-loss eh_wiou `
    --h-wiou-sigma-0 8.0 `
    --seed 42 `
    --use-homotopy-roi `
    --use-egm `
    --epochs 12 `
    --tag "ehwiou_upgraded_s42"
```
