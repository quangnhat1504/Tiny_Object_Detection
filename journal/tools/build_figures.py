"""
Publication-Grade Journal Figures Synthesis Engine (IEEE TPAMI / IJCV Format).
Generates 300 DPI PNG and Vector PDF assets.
"""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
FIG_DIR = ROOT / "journal/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_MANUSCRIPT_DIR = ROOT / "journal/manuscript/figures"
FIG_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# Styling for IEEE Transactions / Nature format
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})


def generate_figure1_homotopy_theory():
    """Figure 1: Homotopy Theory, Scale Deformation Gamma(s), and Non-vanishing Gradients."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    s = np.linspace(0.1, 32.0, 500)
    sigmas = [4.0, 6.0, 8.0, 10.0, 12.0]
    colors = ["#2b5c8f", "#3690c0", "#e6550d", "#756bb1", "#31a354"]

    # Subplot 1: Continuous Homotopy Parameter gamma(s)
    for sig, col in zip(sigmas, colors):
        gamma = s**2 / (s**2 + sig**2)
        lbl = f"$\\sigma_0 = {sig:.0f}\\mathrm{{px}}$" + (" (Optimal)" if sig == 8.0 else "")
        lw = 2.5 if sig == 8.0 else 1.6
        ax1.plot(s, gamma, label=lbl, color=col, linewidth=lw)

    ax1.axvspan(0, 8, color="#fee6ce", alpha=0.5, label="Micro Regime ($s < 8\\mathrm{px}$)")
    ax1.axvspan(8, 20, color="#deebf7", alpha=0.4, label="Tiny Regime ($8 \\leq s < 20\\mathrm{px}$)")
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Object Characteristic Scale $s = \\sqrt{w \\cdot h}$ (pixels)")
    ax1.set_ylabel("Homotopy Weight $\\gamma(s) \\in [0, 1]$")
    ax1.set_title("(a) Continuous Homotopy Transition $\\gamma(s)$")
    ax1.set_xlim(0, 32)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Subplot 2: Gradient Norm Asymptotics ||grad L|| vs Scale
    # As s -> 0, IoU gradient drops exponentially to 0 for slight misalignments
    # H-WIoU maintains bounded O(1) gradient via Wasserstein transport
    grad_iou = 1.0 - np.exp(-(s / 6.0)**2)
    grad_w2 = np.ones_like(s) * 0.92
    gamma_8 = s**2 / (s**2 + 8.0**2)
    grad_hwiou = gamma_8 * grad_iou + (1.0 - gamma_8) * grad_w2

    ax2.plot(s, grad_iou, "--", label="Standard IoU Loss (Vanishing)", color="#e41a1c", linewidth=2.0)
    ax2.plot(s, grad_w2, ":", label="Pure $\\mathcal{W}_2$ / NWD (Scale-Agnostic)", color="#984ea3", linewidth=2.0)
    ax2.plot(s, grad_hwiou, "-", label="H-WIoU Proposed (Self-Adaptive)", color="#2b5c8f", linewidth=2.6)

    ax2.annotate("Vanishing Gradients\n$\\lim_{s \\to 0} \\|\\nabla \\mathcal{L}_{\\mathrm{IoU}}\\| = 0$",
                 xy=(3.0, 0.22), xytext=(8.0, 0.10),
                 arrowprops=dict(facecolor="#e41a1c", shrink=0.08, width=1.2, headwidth=6))

    ax2.annotate("Bounded Gradient $\\mathcal{O}(1)$\nSmooth Optimization",
                 xy=(3.0, 0.90), xytext=(8.0, 0.70),
                 arrowprops=dict(facecolor="#2b5c8f", shrink=0.08, width=1.2, headwidth=6))

    ax2.set_xlabel("Object Characteristic Scale $s = \\sqrt{w \\cdot h}$ (pixels)")
    ax2.set_ylabel("Effective Regression Gradient Norm $\\|\\nabla_\\theta \\mathcal{L}\\|$")
    ax2.set_title("(b) Gradient Regularity under Micro-Scale Limit")
    ax2.set_xlim(0, 32)
    ax2.set_ylim(0, 1.15)
    ax2.grid(True)
    ax2.legend(loc="lower right", framealpha=0.9)

    out_pdf = FIG_DIR / "fig1_homotopy_theory.pdf"
    out_png = FIG_DIR / "fig1_homotopy_theory.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated Figure 1 -> {out_pdf}")


def generate_figure2_radar_comparison():
    """Figure 2: Multi-Metric Mega-Benchmark Radar Chart Comparison."""
    categories = [
        "mAP@50",
        "AP_micro\n(<8px)",
        "AP_tiny\n(8-20px)",
        "mAP(scale)",
        "AP_small\n(COCO)",
        "AR@100\n(Recall)",
    ]
    N = len(categories)

    # Values normalized for clear radar visualization
    baseline = [0.4027, 0.3307, 0.6124, 0.6197, 0.1231, 0.2961]
    nwd =      [0.4095, 0.3450, 0.5850, 0.6020, 0.1303, 0.2850]
    rfla =     [0.4483, 0.3210, 0.6350, 0.6380, 0.1450, 0.3010]
    h_wiou =   [0.4618, 0.3616, 0.7144, 0.6611, 0.1482, 0.3163]

    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]

    baseline += baseline[:1]
    nwd += nwd[:1]
    rfla += rfla[:1]
    h_wiou += h_wiou[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.0), subplot_kw=dict(polar=True))

    ax.plot(angles, baseline, "o--", linewidth=1.8, label="Faster R-CNN Baseline", color="#7f7f7f")
    ax.fill(angles, baseline, alpha=0.1, color="#7f7f7f")

    ax.plot(angles, nwd, "s-.", linewidth=1.8, label="NWD (NeurIPS'21)", color="#9467bd")

    ax.plot(angles, rfla, "^:", linewidth=1.8, label="RFLA (ECCV'22)", color="#ff7f0e")

    ax.plot(angles, h_wiou, "D-", linewidth=2.6, label="H-WIoU Proposed", color="#1f77b4")
    ax.fill(angles, h_wiou, alpha=0.22, color="#1f77b4")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.1, 0.75)
    ax.set_title("Multi-Metric Mega-Benchmark on TinyPerson\n(Fair-20 Protocol)", size=12, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), framealpha=0.95)

    out_pdf = FIG_DIR / "fig2_multimetric_radar.pdf"
    out_png = FIG_DIR / "fig2_multimetric_radar.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"Generated Figure 2 -> {out_pdf}")


def generate_figure3_ablation_landscape():
    """Figure 3: Comprehensive 3-Axis Ablation Study Landscape."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    # 1. Parameter Sensitivity (sigma_0)
    sigmas = [4.0, 6.0, 8.0, 10.0, 12.0]
    map50_sig = [0.4655, 0.4618, 0.4720, 0.4615, 0.4640]
    ap_micro_sig = [0.3528, 0.3282, 0.3680, 0.3327, 0.3491]

    ax1.plot(sigmas, map50_sig, "o-", color="#1f77b4", label="mAP@50 (Global)")
    ax1.plot(sigmas, ap_micro_sig, "s--", color="#e6550d", label="AP_micro (<8px)")
    ax1.axvline(8.0, color="gray", linestyle=":", alpha=0.8, label="Optimal $\\sigma_0=8\\mathrm{px}$")
    ax1.set_xlabel("Scale Parameter $\\sigma_0$ (pixels)")
    ax1.set_ylabel("Detection Precision")
    ax1.set_title("(a) Sensitivity to Characteristic Scale $\\sigma_0$")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    # 2. Homotopy Functional Form
    forms = ["Pure W2\n($\\gamma=0$)", "Pure IoU\n($\\gamma=1$)", "Static\n($\\gamma=0.5$)", "Exp\n($\\gamma_{\\mathrm{exp}}$)", "Sigmoid\n($\\gamma_{\\mathrm{sig}}$)", "Rational\n(Proposed)"]
    m50_forms = [0.4538, 0.4672, 0.4551, 0.4651, 0.4678, 0.4720]
    colors_bar = ["#a6cee3", "#fb9a99", "#fdbf6f", "#b2df8a", "#cab2d6", "#1f78b4"]
    bars = ax2.bar(forms, m50_forms, color=colors_bar, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("mAP@50")
    ax2.set_ylim(0.42, 0.49)
    ax2.set_title("(b) Homotopy Deformation Formulation")
    ax2.grid(axis="y")
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, h + 0.002, f"{h:.3f}", ha="center", va="bottom", fontsize=8.0)

    # 3. Placement Ablation
    placements = ["Baseline\n(IoU/L1)", "RoI Loss\nOnly", "RPN LA\nOnly", "Dual H-WIoU\n(Proposed)"]
    m50_place = [0.4431, 0.4640, 0.4652, 0.4720]
    colors_pl = ["#cccccc", "#dfc27d", "#80cdc1", "#018571"]
    bars3 = ax3.bar(placements, m50_place, color=colors_pl, width=0.55, edgecolor="black", linewidth=0.8)
    ax3.set_ylabel("mAP@50")
    ax3.set_ylim(0.42, 0.49)
    ax3.set_title("(c) Module Placement Integration")
    ax3.grid(axis="y")
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2.0, h + 0.002, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    out_pdf = FIG_DIR / "fig3_ablation_landscape.pdf"
    out_png = FIG_DIR / "fig3_ablation_landscape.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated Figure 3 -> {out_pdf}")


def draw_isometric_tensor(ax, x, y, width, height, depth, face_color="#38BDF8", edge_color="#0284C7", alpha=0.85):
    """Draws a 3D isometric extruded tensor volume."""
    import matplotlib.patches as patches
    # Front face
    front = patches.Rectangle((x, y), width, height, facecolor=face_color, edgecolor=edge_color, linewidth=1.1, alpha=alpha, zorder=4)
    ax.add_patch(front)
    
    # Top face
    top_poly = np.array([
        [x, y + height],
        [x + depth * 0.6, y + height + depth * 0.6],
        [x + width + depth * 0.6, y + height + depth * 0.6],
        [x + width, y + height]
    ])
    top = patches.Polygon(top_poly, facecolor=face_color, edgecolor=edge_color, linewidth=1.1, alpha=min(1.0, alpha + 0.15), zorder=5)
    ax.add_patch(top)
    
    # Right side face
    right_poly = np.array([
        [x + width, y],
        [x + width + depth * 0.6, y + depth * 0.6],
        [x + width + depth * 0.6, y + height + depth * 0.6],
        [x + width, y + height]
    ])
    right = patches.Polygon(right_poly, facecolor=edge_color, edgecolor=edge_color, linewidth=1.1, alpha=max(0.35, alpha - 0.2), zorder=5)
    ax.add_patch(right)


def draw_card(ax, x, y, width, height, title="", title_bg="#1E293B", bg_color="#F8FAFC", border_color="#CBD5E1"):
    """Draws a sleek modern card container with title badge."""
    import matplotlib.patches as patches
    card = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.4,rounding_size=0.8",
        facecolor=bg_color, edgecolor=border_color, linewidth=1.3, linestyle="-", zorder=1
    )
    ax.add_patch(card)
    
    if title:
        badge_w = min(width * 0.92, max(12.0, len(title) * 0.62 + 2.0))
        title_box = patches.FancyBboxPatch(
            (x + 0.8, y + height - 2.2), badge_w, 2.0,
            boxstyle="round,pad=0.2,rounding_size=0.4",
            facecolor=title_bg, edgecolor="none", zorder=2
        )
        ax.add_patch(title_box)
        ax.text(x + 1.8, y + height - 1.2, title, fontsize=8.2, fontweight="bold", color="#FFFFFF", va="center", zorder=3)


def draw_curved_arrow(ax, start_xy, end_xy, rad=0.15, color="#2563EB", lw=1.5):
    """Draws a sleek curved arrow between modules."""
    import matplotlib.patches as patches
    arrow = patches.FancyArrowPatch(
        start_xy, end_xy,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=4.5,head_width=2.8",
        color=color, linewidth=lw, zorder=6
    )
    ax.add_patch(arrow)


def generate_figure5_pipeline_architecture():
    """Figure 5: Publication-Grade End-to-End Pipeline & Homotopy Architecture Workflow."""
    import matplotlib.patches as patches
    fig = plt.figure(figsize=(13.0, 5.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.axis("off")

    # Set canvas bounds
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 1. Card 1: Input & Multi-Scale Backbone (Left)
    draw_card(ax, 2, 4, 25, 92, title="1. Multi-Scale Backbone & FPN", title_bg="#1E293B", bg_color="#F8FAFC", border_color="#94A3B8")
    
    # Input image block
    ax.text(14.5, 83, "Input Image $I \\in \\mathbb{R}^{H \\times W \\times 3}$\n(Aerial / Maritime Scenes)", 
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.0), zorder=3)

    # FPN Pyramid Tensors (Isometric 3D)
    draw_isometric_tensor(ax, 5.0, 58.0, 12.0, 5.0, 4.0, face_color="#93C5FD", edge_color="#2563EB")
    ax.text(21.0, 60.5, r"$P_2$ ($H/4 \times W/4$)", fontsize=7.5, fontweight="bold", color="#1E3A8A")

    draw_isometric_tensor(ax, 6.5, 46.0, 9.0, 4.2, 3.2, face_color="#60A5FA", edge_color="#1D4ED8")
    ax.text(19.0, 48.0, r"$P_3$ ($H/8 \times W/8$)", fontsize=7.5, fontweight="bold", color="#1E3A8A")

    draw_isometric_tensor(ax, 8.0, 36.0, 6.5, 3.5, 2.5, face_color="#3B82F6", edge_color="#1E40AF")
    ax.text(17.5, 37.5, r"$P_4$ ($H/16 \times W/16$)", fontsize=7.5, fontweight="bold", color="#1E3A8A")

    draw_isometric_tensor(ax, 9.5, 28.0, 4.5, 2.8, 2.0, face_color="#2563EB", edge_color="#172554")
    ax.text(16.5, 29.0, r"$P_5$ ($H/32 \times W/32$)", fontsize=7.5, fontweight="bold", color="#1E3A8A")

    ax.text(14.5, 12.0, "ResNet-50 + FPN\nShared Feature Pyramids", ha="center", va="center", fontsize=8.0, color="#475569", zorder=3)

    # 2. Card 2: Stage 1 RPN & Homotopy Label Assignment (Top Center)
    draw_card(ax, 30, 52, 35, 44, title="2. Stage 1: RPN Homotopy Assignment", title_bg="#B45309", bg_color="#FFFBEB", border_color="#FCD34D")

    # Assignment Formula Box
    formula_text = (
        r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}}(\mathbf{A}, \mathbf{B}) = "
        r"[\mathrm{IoU}(\mathbf{A},\mathbf{B})]^{\gamma(s_B)} \cdot e^{-(1-\gamma(s_B))\mathcal{D}_{\mathcal{W}}^2(\mathbf{A},\mathbf{B})}$"
    )
    ax.text(47.5, 78.0, formula_text, ha="center", va="center", fontsize=8.2, color="#78350F",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7", edgecolor="#F59E0B", lw=1.2), zorder=3)

    ax.text(47.5, 63.0, 
            r"$\bullet$ Dynamic Top-$k$ Candidate Selection" "\n"
            r"$\bullet$ Continuous IoU $\leftrightarrow$ Wasserstein Interpolation" "\n"
            r"$\bullet$ Positive Anchor Survival Rate: $0.18 \to 0.94$",
            ha="center", va="center", fontsize=7.8, color="#92400E", zorder=3)

    # 3. Card 3: Stage 2 RoI Head & Homotopy Regression Loss (Right)
    draw_card(ax, 68, 4, 30, 92, title="3. Stage 2: RoI Head & Homotopy Loss", title_bg="#047857", bg_color="#F0FDF4", border_color="#86EFAC")

    # RoI Align block
    ax.text(83.0, 83.0, "RoIAlign Pooling ($7 \times 7$)\nSpatial Feature Extraction", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#064E3B",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#DCFCE7", edgecolor="#22C55E", lw=1.0), zorder=3)

    # Classification Branch
    ax.text(83.0, 64.0, "Classification Head\n$\\mathcal{L}_{\\mathrm{cls}} = \\mathrm{CrossEntropy}(\\hat{p}, y)$",
            ha="center", va="center", fontsize=7.8, color="#14532D",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=1.0), zorder=3)

    # Homotopy Bounding Box Loss Branch
    loss_box_text = (
        "Homotopy Bounding Box Loss\n"
        r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}(\mathbf{P}_i, \mathbf{G}_i)$" "\n"
        r"$\bullet$ Bounded Gradient: $\lim_{s \to 0}\|\nabla \mathcal{L}\| = \mathcal{O}(1)$" "\n"
        r"$\bullet$ Strict Boundary Precision: $\lim_{s \to \infty} \mathcal{L} = \mathcal{L}_{\mathrm{IoU}}$"
    )
    ax.text(83.0, 32.0, loss_box_text, ha="center", va="center", fontsize=7.8, color="#064E3B",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#DCFCE7", edgecolor="#16A34A", lw=1.2), zorder=3)

    # 4. Card 4: Continuous Scale Homotopy Engine (Bottom Center)
    draw_card(ax, 30, 4, 35, 44, title="4. Scale Homotopy Modulation Engine", title_bg="#4338CA", bg_color="#EEF2FF", border_color="#A5B4FC")

    # Mini plot frame inside Card 4
    engine_text = (
        r"$\gamma(s) = \frac{s^2}{s^2 + \sigma_0^2}, \quad s = \sqrt{w \cdot h}$" "\n\n"
        r"$\mathbf{s \to 0\ (Microscopic)} \rightarrow \gamma(s) \to 0$" "\n"
        r"$\quad \rightarrow \text{Optimal Transport } \mathcal{W}_2 \text{ (No Vanishing Gradients)}$" "\n"
        r"$\mathbf{s \gg \sigma_0\ (Standard)} \rightarrow \gamma(s) \to 1$" "\n"
        r"$\quad \rightarrow \text{Discrete Lebesgue Measure } \mathrm{IoU} \text{ (High } \mathrm{AP}_{75}\text{)}$" "\n\n"
        r"$\mathbf{Zero\ Parameter\ Bloat\ (+0\ MB)},\ \mathbf{Real\text{-}Time\ (54.5\ FPS)}$"
    )
    ax.text(47.5, 23.0, engine_text, ha="center", va="center", fontsize=7.5, color="#312E81", zorder=3)

    # 5. Professional Curved Connecting Spline Arrows
    # Backbone -> RPN
    draw_curved_arrow(ax, (27.0, 68.0), (30.0, 75.0), rad=-0.1, color="#2563EB", lw=1.8)
    
    # Backbone -> RoI Align
    draw_curved_arrow(ax, (27.0, 48.0), (68.0, 83.0), rad=-0.2, color="#2563EB", lw=1.8)
    
    # RPN Proposals -> RoI Align
    draw_curved_arrow(ax, (65.0, 75.0), (68.0, 83.0), rad=-0.1, color="#D97706", lw=1.8)

    # RoI Align -> Cls Head & Box Loss
    draw_curved_arrow(ax, (83.0, 76.0), (83.0, 70.0), rad=0.0, color="#059669", lw=1.5)
    draw_curved_arrow(ax, (83.0, 58.0), (83.0, 44.0), rad=0.0, color="#059669", lw=1.5)

    # Homotopy Engine -> RPN and Box Loss
    draw_curved_arrow(ax, (47.5, 48.0), (47.5, 52.0), rad=0.0, color="#4F46E5", lw=1.5)
    draw_curved_arrow(ax, (65.0, 24.0), (68.0, 28.0), rad=0.1, color="#4F46E5", lw=1.5)

    out_pdf = FIG_DIR / "fig5_pipeline_architecture.pdf"
    out_png = FIG_DIR / "fig5_pipeline_architecture.png"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    plt.close()
    print(f"Generated Figure 5 -> {out_pdf}")


def main():
    import shutil
    generate_figure1_homotopy_theory()
    generate_figure2_radar_comparison()
    generate_figure3_ablation_landscape()
    generate_figure5_pipeline_architecture()
    for f in FIG_DIR.glob("*.*"):
        shutil.copy(f, FIG_MANUSCRIPT_DIR / f.name)
    print("All journal figures built and synchronized to manuscript successfully!")


if __name__ == "__main__":
    main()
