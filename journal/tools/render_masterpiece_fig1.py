"""
Masterpiece Academic Figure 1 Generator for IEEE TPAMI.
Built strictly addressing all 4 user feedback axes:
1. Academic Serif/Computer Modern LaTeX Typography & Color-blind friendly Seaborn/Muted palette.
2. (a) Standard IoU Collapse: Spatial Grid + 2D Flat Plateau Loss Surface + Vanishing Gradient vector.
3. (b) Gaussian Wasserstein Space: Multi-ring concentric probability density contours + Curved Optimal Transport Vector.
4. (c) Continuous Homotopy Deformation: Quantitative 2D Line Chart of gamma(d) with shaded regime boundaries & asymptotic convergence.
"""
from __future__ import annotations
import math
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Polygon, FancyArrowPatch

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
OUT_DIR = ROOT / "journal/manuscript/figures"
FIG_DIR = ROOT / "journal/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Strict Academic LaTeX Typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
    "text.usetex": False,  # Robust mathtext parser
})

# Color-blind friendly academic palette (Navy, Bordeaux, Sage, Amber, Slate)
NAVY_DEEP   = "#1E3A8A"  # GT Box / IoU dominant
BORDEAUX    = "#881337"  # Anchor Box / Collapse regime
BORDEAUX_BG = "#FFF1F2"
TEAL_DEEP   = "#0F766E"  # Gaussian density / Optimal Transport
TEAL_BG     = "#F0FDF4"
AMBER_DEEP  = "#B45309"  # Proposed Homotopy
AMBER_BG    = "#FFFBEB"
SLATE_DARK  = "#0F172A"
SLATE_MUTED = "#64748B"
BORDER_GRAY = "#CBD5E1"


def render_masterpiece_figure1():
    print("Generating Masterpiece Academic Figure 1...")
    fig = plt.figure(figsize=(11.8, 4.3), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    # 3 Column Subplots
    # Left: IoU Collapse (Grid + Boxes + Flat Surface)
    # Mid:  Gaussian Wasserstein (Concentric Contours + Transport Vector + Smooth Surface)
    # Right: Homotopy Transition Curve (2D gamma(d) plot + regimes)
    
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.22, left=0.04, right=0.97, top=0.88, bottom=0.10)
    
    # --------------------------------------------------------------------------
    # COLUMN 1: (a) Standard IoU Collapse
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-1, 10)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_aspect("equal")
    ax1.set_facecolor("#F8FAFC")
    
    # Border card styling
    for spine in ax1.spines.values():
        spine.set_color(BORDER_GRAY)
        spine.set_linewidth(1.0)

    # Title Banner
    ax1.set_title(r"$\mathbf{(a)\ Standard\ IoU\ Metric\ Collapse}$", fontsize=9.5, pad=10, color=BORDEAUX, weight="bold")

    # Spatial Coordinate Grid
    for g in range(0, 10):
        ax1.axvline(g, color="#E2E8F0", lw=0.6, zorder=1)
        ax1.axhline(g, color="#E2E8F0", lw=0.6, zorder=1)

    # Anchor Box A (8x8)
    rect_a = patches.Rectangle((1.0, 4.0), 3.0, 3.0, linewidth=1.6, edgecolor=BORDEAUX, facecolor="#FECDD3", alpha=0.6, zorder=3)
    ax1.add_patch(rect_a)
    ax1.text(2.5, 5.5, r"$\mathbf{A}$", ha="center", va="center", fontsize=9.0, color=BORDEAUX, weight="bold", zorder=4)
    ax1.text(2.5, 7.3, r"$8\times 8\,\mathrm{px}$", ha="center", va="center", fontsize=7.0, color=BORDEAUX, zorder=4)

    # Ground Truth Box G (6x6)
    rect_g = patches.Rectangle((6.0, 4.5), 2.5, 2.5, linewidth=1.6, edgecolor=NAVY_DEEP, facecolor="#BFDBFE", alpha=0.6, zorder=3)
    ax1.add_patch(rect_g)
    ax1.text(7.25, 5.75, r"$\mathbf{G}$", ha="center", va="center", fontsize=9.0, color=NAVY_DEEP, weight="bold", zorder=4)
    ax1.text(7.25, 7.3, r"$6\times 6\,\mathrm{px}$", ha="center", va="center", fontsize=7.0, color=NAVY_DEEP, zorder=4)

    # Vanishing Gap & Gradient Indicator
    ax1.annotate("", xy=(6.0, 5.75), xytext=(4.0, 5.75),
                 arrowprops=dict(arrowstyle="<->", color="#991B1B", lw=1.2, shrinkA=0, shrinkB=0), zorder=4)
    ax1.text(5.0, 6.2, r"$\Delta x > 0$", ha="center", va="bottom", fontsize=7.5, color=BORDEAUX, weight="bold", zorder=5)
    
    # Prominent Vanishing Gradient callout
    bbox_props = dict(boxstyle="round,pad=0.3", fc="#FEF2F2", ec=BORDEAUX, lw=1.0)
    ax1.text(5.0, 4.0, r"$\mathrm{IoU}(A,G) = 0$" + "\n" + r"$\nabla_{\!A}\,\mathcal{L}_{\mathrm{IoU}} = \mathbf{0}$",
             ha="center", va="top", fontsize=8.0, color=BORDEAUX, bbox=bbox_props, zorder=5)

    # Inset / Bottom: Flat Plateau Loss Surface Illustration
    ax1_sub = ax1.inset_axes([0.08, 0.04, 0.84, 0.28])
    dx = np.linspace(-3, 6, 200)
    loss_iou = np.where(dx <= 0, 1.0 - (1.0 + dx/3.0), 1.0)
    loss_iou = np.clip(loss_iou, 0.0, 1.0)
    
    ax1_sub.plot(dx, loss_iou, color=BORDEAUX, lw=1.8, zorder=3)
    ax1_sub.fill_between(dx[dx >= 0], 0, 1.0, color="#FECDD3", alpha=0.35, zorder=2)
    ax1_sub.axvline(0, color=SLATE_MUTED, ls=":", lw=0.8)
    ax1_sub.text(2.5, 0.5, r"$\mathrm{Flat\ Plateau}\ (\nabla = \mathbf{0})$", ha="center", va="center", fontsize=6.8, color=BORDEAUX, style="italic")
    ax1_sub.set_xlim(-3, 6)
    ax1_sub.set_ylim(-0.1, 1.2)
    ax1_sub.set_xticks([])
    ax1_sub.set_yticks([0, 1])
    ax1_sub.set_yticklabels(["0", "1"], fontsize=6.5, color=SLATE_MUTED)
    ax1_sub.set_ylabel(r"$\mathcal{L}_{\mathrm{IoU}}$", fontsize=7.2, color=SLATE_DARK, labelpad=-2)
    ax1_sub.set_facecolor("#FFFFFF")
    for s in ax1_sub.spines.values():
        s.set_color(BORDER_GRAY)
        s.set_linewidth(0.7)

    ax1.set_xticks([])
    ax1.set_yticks([])

    # --------------------------------------------------------------------------
    # COLUMN 2: (b) Bivariate Gaussian Wasserstein Space
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(-1, 10)
    ax2.set_ylim(-0.5, 9.5)
    ax2.set_aspect("equal")
    ax2.set_facecolor("#F8FAFC")
    
    for spine in ax2.spines.values():
        spine.set_color(BORDER_GRAY)
        spine.set_linewidth(1.0)

    # Title Banner
    ax2.set_title(r"$\mathbf{(b)\ Gaussian\ Optimal\ Transport}$", fontsize=9.5, pad=10, color=TEAL_DEEP, weight="bold")

    for g in range(0, 10):
        ax2.axvline(g, color="#E2E8F0", lw=0.6, zorder=1)
        ax2.axhline(g, color="#E2E8F0", lw=0.6, zorder=1)

    # Anchor Gaussian Density Contours (Concentric Fading Ellipses)
    center_a = (2.5, 5.5)
    for rad_scale, alpha, ls in [(3.2, 0.15, ":"), (2.2, 0.30, "--"), (1.2, 0.60, "-")]:
        e = Ellipse(center_a, 2.8 * rad_scale / 2.0, 2.4 * rad_scale / 2.0, angle=20,
                    edgecolor=TEAL_DEEP, facecolor="#A7F3D0", alpha=alpha, linestyle=ls, lw=1.0, zorder=2)
        ax2.add_patch(e)
    ax2.plot(center_a[0], center_a[1], marker="o", color=TEAL_DEEP, markersize=3.5, zorder=5)
    ax2.text(center_a[0], center_a[1] - 0.45, r"$\mu_A$", ha="center", va="top", fontsize=7.5, color=TEAL_DEEP, weight="bold", zorder=5)
    ax2.text(center_a[0], 7.5, r"$\mathcal{N}_A(\mu_A, \Sigma_A)$", ha="center", va="center", fontsize=7.2, color=TEAL_DEEP, zorder=5)

    # GT Gaussian Density Contours
    center_g = (7.5, 5.5)
    for rad_scale, alpha, ls in [(2.8, 0.15, ":"), (1.9, 0.30, "--"), (1.0, 0.60, "-")]:
        e = Ellipse(center_g, 2.4 * rad_scale / 2.0, 2.0 * rad_scale / 2.0, angle=-10,
                    edgecolor=NAVY_DEEP, facecolor="#BFDBFE", alpha=alpha, linestyle=ls, lw=1.0, zorder=2)
        ax2.add_patch(e)
    ax2.plot(center_g[0], center_g[1], marker="o", color=NAVY_DEEP, markersize=3.5, zorder=5)
    ax2.text(center_g[0], center_g[1] - 0.45, r"$\mu_G$", ha="center", va="top", fontsize=7.5, color=NAVY_DEEP, weight="bold", zorder=5)
    ax2.text(center_g[0], 7.5, r"$\mathcal{N}_G(\mu_G, \Sigma_G)$", ha="center", va="center", fontsize=7.2, color=NAVY_DEEP, zorder=5)

    # Curved Optimal Transport Vector connecting mu_A and mu_G
    arrow_arc = patches.FancyArrowPatch(
        center_a, center_g,
        connectionstyle="arc3,rad=-0.22",
        arrowstyle="-|>,head_length=5,head_width=3",
        color=TEAL_DEEP, lw=1.8, ls="--", zorder=6
    )
    ax2.add_patch(arrow_arc)
    ax2.text(5.0, 6.7, r"$\mathbf{W_2(\mathcal{N}_A, \mathcal{N}_G)}$", ha="center", va="bottom", fontsize=8.0, color=TEAL_DEEP, weight="bold", zorder=6)

    # Continuous Formula Callout
    bbox_teal = dict(boxstyle="round,pad=0.3", fc="#F0FDF4", ec=TEAL_DEEP, lw=1.0)
    ax2.text(5.0, 4.0, r"$\mathrm{NWD} = \exp\left(-\frac{W_2}{C}\right) > 0$" + "\n" + r"$\nabla_{\!A}\,\mathcal{L}_{\mathrm{NWD}} \neq \mathbf{0}\quad (\forall\, d)$",
             ha="center", va="top", fontsize=8.0, color=TEAL_DEEP, bbox=bbox_teal, zorder=5)

    # Inset / Bottom: Smooth Convex Loss Surface
    ax2_sub = ax2.inset_axes([0.08, 0.04, 0.84, 0.28])
    dx = np.linspace(-3, 6, 200)
    loss_nwd = 1.0 - np.exp(-np.abs(dx)/2.5)
    
    ax2_sub.plot(dx, loss_nwd, color=TEAL_DEEP, lw=1.8, zorder=3)
    ax2_sub.fill_between(dx, 0, loss_nwd, color="#A7F3D0", alpha=0.35, zorder=2)
    ax2_sub.axvline(0, color=SLATE_MUTED, ls=":", lw=0.8)
    ax2_sub.text(2.2, 0.35, r"$\mathrm{Smooth\ Gradient}\ (\forall\, d)$", ha="center", va="center", fontsize=6.8, color=TEAL_DEEP, style="italic")
    ax2_sub.set_xlim(-3, 6)
    ax2_sub.set_ylim(-0.1, 1.2)
    ax2_sub.set_xticks([])
    ax2_sub.set_yticks([0, 1])
    ax2_sub.set_yticklabels(["0", "1"], fontsize=6.5, color=SLATE_MUTED)
    ax2_sub.set_ylabel(r"$\mathcal{L}_{\mathrm{NWD}}$", fontsize=7.2, color=SLATE_DARK, labelpad=-2)
    ax2_sub.set_facecolor("#FFFFFF")
    for s in ax2_sub.spines.values():
        s.set_color(BORDER_GRAY)
        s.set_linewidth(0.7)

    ax2.set_xticks([])
    ax2.set_yticks([])

    # --------------------------------------------------------------------------
    # COLUMN 3: (c) Quantitative Continuous Homotopy Weighting Curve
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#FFFFFF")
    
    for spine in ax3.spines.values():
        spine.set_color(BORDER_GRAY)
        spine.set_linewidth(1.0)

    # Title Banner
    ax3.set_title(r"$\mathbf{(c)\ Continuous\ Homotopy\ Deformation}$", fontsize=9.5, pad=10, color=AMBER_DEEP, weight="bold")

    d_vals = np.linspace(0, 36, 300)
    sigma_0 = 8.0
    gamma_vals = 1.0 / (1.0 + (d_vals / sigma_0)**2)

    # Shaded Regime Bands
    # 1. Tiny Scale Regime (d <= 8)
    ax3.axvspan(0, 8, color="#FEF3C7", alpha=0.5, label=r"$\mathrm{Tiny\ Regime}\ (\gamma \to 1)$", zorder=1)
    # 2. Transition Scale Regime (8 < d <= 20)
    ax3.axvspan(8, 20, color="#F1F5F9", alpha=0.6, label=r"$\mathrm{Transition\ Regime}$", zorder=1)
    # 3. Standard Scale Regime (d > 20)
    ax3.axvspan(20, 36, color="#DBEAFE", alpha=0.35, label=r"$\mathrm{Standard\ Regime}\ (\gamma \to 0)$", zorder=1)

    # Main Curve
    ax3.plot(d_vals, gamma_vals, color=AMBER_DEEP, lw=2.4, zorder=4, label=r"$\gamma(d) = \frac{1}{1 + (d/\sigma_0)^2}$")

    # Critical Point Markers
    # d = 0 -> gamma = 1.0 (Pure NWD)
    ax3.plot(0, 1.0, marker="o", color=TEAL_DEEP, markersize=5.5, zorder=5)
    ax3.text(1.2, 0.98, r"$\gamma(0)=1.0\ (\mathrm{NWD})$", fontsize=7.2, color=TEAL_DEEP, weight="bold")

    # d = sigma_0 = 8.0 -> gamma = 0.5 (Equilibrium)
    ax3.plot(8.0, 0.5, marker="s", color=AMBER_DEEP, markersize=6.0, zorder=5)
    ax3.vlines(8.0, 0, 0.5, color=AMBER_DEEP, linestyle=":", lw=1.0, zorder=3)
    ax3.hlines(0.5, 0, 8.0, color=AMBER_DEEP, linestyle=":", lw=1.0, zorder=3)
    ax3.text(8.8, 0.52, r"$\sigma_0=8\mathrm{px}\ (\gamma=0.5)$", fontsize=7.2, color=AMBER_DEEP, weight="bold")

    # d -> infty -> gamma = 0.0 (Pure IoU)
    ax3.plot(32.0, 1.0/(1+(32/8)**2), marker="^", color=NAVY_DEEP, markersize=5.5, zorder=5)
    ax3.text(22.0, 0.15, r"$\gamma \to 0\ (\mathrm{IoU})$", fontsize=7.2, color=NAVY_DEEP, weight="bold")

    # Homotopy Convex Combination Formula Callout Box
    ax3.text(18.0, 0.82,
             r"$H_\gamma(A, G) = (1 - \gamma)\mathrm{IoU} + \gamma \mathrm{NWD}$" + "\n" +
             r"$\bullet\ \text{Strictly Smooth C}^\infty\ \text{Deformation}$" + "\n" +
             r"$\bullet\ \text{Zero Gradient Discontinuity}$",
             ha="center", va="center", fontsize=7.2, color=SLATE_DARK,
             bbox=dict(boxstyle="round,pad=0.35", fc="#FFFBEB", ec=AMBER_DEEP, lw=1.0), zorder=5)

    ax3.set_xlim(0, 36)
    ax3.set_ylim(-0.02, 1.08)
    ax3.set_xlabel(r"$\mathrm{Geometric\ Scale}\ d = \sqrt{w \cdot h}\ (\mathrm{pixels})$", fontsize=7.8, color=SLATE_DARK, labelpad=4)
    ax3.set_ylabel(r"$\mathrm{Homotopy\ Weight}\ \gamma(d)$", fontsize=7.8, color=SLATE_DARK, labelpad=2)
    ax3.set_xticks([0, 8, 16, 24, 32])
    ax3.set_xticklabels(["0", r"$\sigma_0$", "16", "24", "32"], fontsize=7.2, color=SLATE_DARK)
    ax3.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=7.0, color=SLATE_DARK)
    ax3.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0", zorder=0)

    # Save to both paths
    pdf_out1 = OUT_DIR / "fig1_homotopy_theory.pdf"
    png_out1 = OUT_DIR / "fig1_homotopy_theory.png"
    pdf_out2 = FIG_DIR / "fig1_homotopy_theory.pdf"
    png_out2 = FIG_DIR / "fig1_homotopy_theory.png"

    plt.savefig(pdf_out1, format="pdf", bbox_inches="tight")
    plt.savefig(png_out1, format="png", dpi=300, bbox_inches="tight")
    plt.savefig(pdf_out2, format="pdf", bbox_inches="tight")
    plt.savefig(png_out2, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Masterpiece Figure 1 successfully generated at:\n  -> {pdf_out1}\n  -> {png_out1}")

if __name__ == "__main__":
    render_masterpiece_figure1()
