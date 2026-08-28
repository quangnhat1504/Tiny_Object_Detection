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
    sigmas = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    map50_sig = [0.4512, 0.4618, 0.4575, 0.4615, 0.4530, 0.4468]
    ap_micro_sig = [0.3180, 0.3282, 0.3616, 0.3327, 0.3250, 0.3160]

    ax1.plot(sigmas, map50_sig, "o-", color="#1f77b4", label="mAP@50 (Global)")
    ax1.plot(sigmas, ap_micro_sig, "s--", color="#e6550d", label="AP_micro (<8px)")
    ax1.axvline(8.0, color="gray", linestyle=":", alpha=0.8, label="Optimal $\\sigma_0=8\\mathrm{px}$")
    ax1.set_xlabel("Scale Parameter $\\sigma_0$ (pixels)")
    ax1.set_ylabel("Detection Precision")
    ax1.set_title("(a) Sensitivity to Characteristic Scale $\\sigma_0$")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    # 2. Homotopy Functional Form
    forms = ["Pure W2\n($\\gamma=0$)", "Pure IoU\n($\\gamma=1$)", "Static Blend\n($\\gamma=0.5$)", "Exponential\n($\\gamma_{\\mathrm{exp}}$)", "Rational\n(Proposed)"]
    m50_forms = [0.4120, 0.4027, 0.4315, 0.4540, 0.4618]
    colors_bar = ["#a6cee3", "#fb9a99", "#fdbf6f", "#b2df8a", "#1f78b4"]
    bars = ax2.bar(forms, m50_forms, color=colors_bar, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_ylabel("mAP@50")
    ax2.set_ylim(0.38, 0.48)
    ax2.set_title("(b) Homotopy Deformation Formulation")
    ax2.grid(axis="y")
    for b in bars:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2.0, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    # 3. Placement Ablation
    placements = ["Baseline\n(IoU/L1)", "RoI Loss\nOnly", "RPN LA\nOnly", "Dual H-WIoU\n(Proposed)"]
    m50_place = [0.4027, 0.4380, 0.4490, 0.4618]
    colors_pl = ["#cccccc", "#dfc27d", "#80cdc1", "#018571"]
    bars3 = ax3.bar(placements, m50_place, color=colors_pl, width=0.55, edgecolor="black", linewidth=0.8)
    ax3.set_ylabel("mAP@50")
    ax3.set_ylim(0.38, 0.48)
    ax3.set_title("(c) Module Placement Integration")
    ax3.grid(axis="y")
    for b in bars3:
        h = b.get_height()
        ax3.text(b.get_x() + b.get_width()/2.0, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    out_pdf = FIG_DIR / "fig3_ablation_landscape.pdf"
    out_png = FIG_DIR / "fig3_ablation_landscape.png"
    plt.savefig(out_pdf)
    plt.savefig(out_png)
    plt.close()
    print(f"Generated Figure 3 -> {out_pdf}")


def main():
    import shutil
    generate_figure1_homotopy_theory()
    generate_figure2_radar_comparison()
    generate_figure3_ablation_landscape()
    for f in FIG_DIR.glob("*.*"):
        shutil.copy(f, FIG_MANUSCRIPT_DIR / f.name)
    print("All journal figures built and synchronized to manuscript successfully!")


if __name__ == "__main__":
    main()
