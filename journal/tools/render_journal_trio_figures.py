"""
Masterpiece Journal Trio Figure Generator for IEEE TPAMI.
Strictly implements the user's 3-figure specification:
1. Figure 1: The Teaser / Motivation Figure (Split-screen: Standard RPN Starvation >70% vs H-WIoU HLA 0.18->0.94 + Gaussian Receptive Field).
2. Figure 2: The Mathematical Intuition (2D Homotopy parameter curve gamma(s), Wasserstein exp(-D_W^2) to IoU asymptotic regimes).
3. Figure 3/5: The Architecture Pipeline (Gaussian embedding, Stage 1 RPN matrix S_ij, Stage 2 Bounded Loss [0, 1], Backprop non-vanishing gradient ||grad|| = O(1) > 0).
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
})

# Academic Muted & Color-blind Friendly Palette
NAVY_DEEP   = "#1E3A8A"  # Primary blue / GT
BORDEAUX    = "#881337"  # Failure / IoU collapse
TEAL_DEEP   = "#0F766E"  # Gaussian transport / Optimal bound
AMBER_DEEP  = "#B45309"  # Proposed Homotopy / Accent
SLATE_DARK  = "#0F172A"
SLATE_MUTED = "#64748B"
BORDER_GRAY = "#CBD5E1"


# ==============================================================================
# FIGURE 1: TEASER / MOTIVATION (SPLIT-SCREEN COMPARISON)
# ==============================================================================
def render_figure1_teaser():
    print("Generating Figure 1 (Motivation & Teaser)...")
    fig = plt.figure(figsize=(11.5, 4.4), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.14, left=0.04, right=0.96, top=0.88, bottom=0.08)

    # --------------------------------------------------------------------------
    # LEFT PANEL: Standard RPN (Anchor Starvation > 70%, Survival = 0.18)
    # --------------------------------------------------------------------------
    ax_left = fig.add_subplot(gs[0, 0])
    ax_left.set_xlim(0, 10)
    ax_left.set_ylim(0, 10)
    ax_left.set_aspect("equal")
    ax_left.set_facecolor("#FEF2F2")

    for s in ax_left.spines.values():
        s.set_color("#FECACA")
        s.set_linewidth(1.2)

    ax_left.set_title(r"$\mathbf{Standard\ RPN:\ Positive\ Anchor\ Starvation\ (>70\%)}$",
                      fontsize=9.0, pad=10, color=BORDEAUX, weight="bold")

    # Spatial Anchor Grid
    for i in range(1, 10):
        ax_left.axvline(i, color="#FCA5A5", lw=0.5, ls=":", alpha=0.6)
        ax_left.axhline(i, color="#FCA5A5", lw=0.5, ls=":", alpha=0.6)

    # Grid Anchor boxes (Starved / Zero IoU)
    anchor_coords = [(2, 2), (2, 5), (2, 8), (5, 2), (5, 8), (8, 2), (8, 5), (8, 8)]
    for (ax_x, ax_y) in anchor_coords:
        ax_left.add_patch(Rectangle((ax_x - 0.9, ax_y - 0.9), 1.8, 1.8, lw=0.9, edgecolor="#F87171", facecolor="none", ls="--", alpha=0.7))

    # Center tiny ground truth (< 8px)
    gt_rect = Rectangle((4.7, 4.7), 0.6, 0.6, lw=1.8, edgecolor=NAVY_DEEP, facecolor="#93C5FD", alpha=0.9, zorder=5)
    ax_left.add_patch(gt_rect)
    ax_left.text(5.0, 4.2, r"$\mathrm{Tiny\ GT}\ (s < 8\mathrm{px})$", ha="center", va="top", fontsize=7.2, color=NAVY_DEEP, weight="bold", zorder=6)

    # Center anchor (barely disjoint)
    cand_anchor = Rectangle((5.2, 5.2), 1.8, 1.8, lw=1.4, edgecolor=BORDEAUX, facecolor="#FCA5A5", alpha=0.4, ls="--", zorder=4)
    ax_left.add_patch(cand_anchor)
    ax_left.text(6.1, 7.3, r"$\mathrm{Anchor}\ A_i$", ha="center", va="center", fontsize=7.0, color=BORDEAUX, zorder=6)

    # Vanishing Gap indicator
    ax_left.annotate("", xy=(5.3, 5.0), xytext=(4.7, 5.0),
                     arrowprops=dict(arrowstyle="<->", color=BORDEAUX, lw=1.2), zorder=6)
    ax_left.text(5.0, 5.5, r"$\mathrm{IoU} < 0.2$", ha="center", va="bottom", fontsize=7.5, color=BORDEAUX, weight="bold", zorder=6)

    # Stats Banner (Bottom Card)
    ax_left.text(5.0, 1.4,
                 r"$\mathbf{Catastrophic\ Failure\ on\ Micro\ Objects:}$" + "\n" +
                 r"$\bullet\ \mathrm{Positive\ Survival\ Rate}:\ \mathbf{0.18}\ (82\%\ \mathrm{Missed})$" + "\n" +
                 r"$\bullet\ \mathrm{Gradient\ Vanishing}:\ \nabla_{\!A}\,\mathcal{L}_{\mathrm{IoU}} = \mathbf{0}\quad (\mathrm{Area} \cap = 0)$",
                 ha="center", va="center", fontsize=7.4, color=BORDEAUX,
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#FECACA", lw=1.0), zorder=7)

    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # --------------------------------------------------------------------------
    # RIGHT PANEL: H-WIoU Stage 1 (HLA: Positive Survival 0.18 -> 0.94 + Gaussian RF)
    # --------------------------------------------------------------------------
    ax_right = fig.add_subplot(gs[0, 1])
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)
    ax_right.set_aspect("equal")
    ax_right.set_facecolor("#F0FDF4")

    for s in ax_right.spines.values():
        s.set_color("#BBF7D0")
        s.set_linewidth(1.2)

    ax_right.set_title(r"$\mathbf{H-WIoU\ Stage\ 1:\ Homotopy\ Assignment\ (Survival\ \mathbf{0.94})}$",
                       fontsize=9.0, pad=10, color=TEAL_DEEP, weight="bold")

    # Spatial Anchor Grid
    for i in range(1, 10):
        ax_right.axvline(i, color="#A7F3D0", lw=0.5, ls=":", alpha=0.6)
        ax_right.axhline(i, color="#A7F3D0", lw=0.5, ls=":", alpha=0.6)

    # Dynamic Receptive Field (Multi-ring Faint Gaussian Density Ellipses)
    center = (5.0, 5.0)
    for scale, alpha, ls in [(4.5, 0.12, ":"), (3.0, 0.22, "--"), (1.8, 0.40, "-")]:
        e = Ellipse(center, scale, scale, edgecolor=TEAL_DEEP, facecolor="#6EE7B7", alpha=alpha, linestyle=ls, lw=1.0, zorder=2)
        ax_right.add_patch(e)

    # Assigned Positive Anchors (Surviving Candidates with Smooth Weight)
    assigned_anchors = [(5, 5), (3.8, 5.0), (6.2, 5.0), (5.0, 3.8), (5.0, 6.2)]
    for (ax_x, ax_y) in assigned_anchors:
        ax_right.add_patch(Rectangle((ax_x - 0.7, ax_y - 0.7), 1.4, 1.4, lw=1.2, edgecolor=TEAL_DEEP, facecolor="#A7F3D0", alpha=0.5, zorder=3))
        ax_right.plot(ax_x, ax_y, marker="o", color=TEAL_DEEP, markersize=3.0, zorder=4)

    # Tiny Ground Truth
    gt_rect2 = Rectangle((4.7, 4.7), 0.6, 0.6, lw=1.8, edgecolor=NAVY_DEEP, facecolor="#93C5FD", alpha=0.9, zorder=5)
    ax_right.add_patch(gt_rect2)
    ax_right.text(5.0, 4.2, r"$\mathrm{Tiny\ GT}\ (s < 8\mathrm{px})$", ha="center", va="top", fontsize=7.2, color=NAVY_DEEP, weight="bold", zorder=6)

    # Dynamic Receptive Field Callout
    ax_right.annotate(r"$\mathrm{Gaussian\ Dynamic\ Field}$", xy=(7.2, 6.8), xytext=(7.2, 8.2),
                      ha="center", fontsize=6.8, color=TEAL_DEEP, weight="bold",
                      arrowprops=dict(arrowstyle="->", color=TEAL_DEEP, lw=1.0), zorder=6)

    # Stats Banner (Bottom Card)
    ax_right.text(5.0, 1.4,
                  r"$\mathbf{Homotopy\ Label\ Assignment\ (HLA)\ Breakthrough:}$" + "\n" +
                  r"$\bullet\ \mathrm{Positive\ Survival\ Rate}:\ \mathbf{0.18 \to 0.94}\ (+422\%\ \mathrm{Gain})$" + "\n" +
                  r"$\bullet\ \mathrm{Continuous\ Optimal\ Transport}:\ \mathcal{S}_{\mathrm{H-WIoU}} > 0\quad (\forall\, A_i)$",
                  ha="center", va="center", fontsize=7.4, color=TEAL_DEEP,
                  bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#BBF7D0", lw=1.0), zorder=7)

    ax_right.set_xticks([])
    ax_right.set_yticks([])

    pdf_out = OUT_DIR / "fig1_homotopy_theory.pdf"
    png_out = OUT_DIR / "fig1_homotopy_theory.png"
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Figure 1 (Teaser) to {pdf_out} and {png_out}")


# ==============================================================================
# FIGURE 2: THE MATHEMATICAL INTUITION (HOMOTOPY CONTINUOUS METRIC SPACE)
# ==============================================================================
def render_figure2_math_intuition():
    print("Generating Figure 2 (Mathematical Intuition)...")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    for s in ax.spines.values():
        s.set_color(BORDER_GRAY)
        s.set_linewidth(1.0)

    # Spatial scale range s from 0 to 40 pixels
    s_vals = np.linspace(0, 40, 400)
    sigma_0 = 8.0
    gamma_s = (s_vals**2) / (s_vals**2 + sigma_0**2)

    # Shaded Regime Bands
    # 1. Micro Scale Regime (Wasserstein Dominant)
    ax.axvspan(0, 8, color="#FEF3C7", alpha=0.5, label=r"$\mathrm{Micro\ Regime}\ (s \leq 8\mathrm{px}):\ \mathcal{S} \to \exp(-\mathcal{D}_{\mathcal{W}}^2)$", zorder=1)
    # 2. Transition Regime
    ax.axvspan(8, 24, color="#F1F5F9", alpha=0.6, label=r"$\mathrm{Transition\ Regime}\ (8 < s \leq 24\mathrm{px})$", zorder=1)
    # 3. Macro Scale Regime (IoU Dominant)
    ax.axvspan(24, 40, color="#DBEAFE", alpha=0.35, label=r"$\mathrm{Macro\ Regime}\ (s > 24\mathrm{px}):\ \mathcal{S} \to \mathrm{IoU}$", zorder=1)

    # Continuous Homotopy Curve
    ax.plot(s_vals, gamma_s, color=AMBER_DEEP, lw=2.6, zorder=4, label=r"$\gamma(s) = \frac{s^2}{s^2 + \sigma_0^2}\quad (\sigma_0 = 8.0\mathrm{px})$")

    # Asymptotic Limits & Markers
    # Limit 1: s -> 0+ (Micro limit)
    ax.plot(0, 0, marker="o", color=TEAL_DEEP, markersize=6.0, zorder=5)
    ax.text(1.2, 0.06, r"$\lim_{s \to 0^+} \gamma(s) = 0 \Rightarrow \mathcal{S}_{\mathrm{H-WIoU}} \to \exp(-\mathcal{D}_{\mathcal{W}}^2)$",
            fontsize=7.8, color=TEAL_DEEP, weight="bold", zorder=6)

    # Critical Point: s = sigma_0 = 8px -> gamma = 0.5 (Equilibrium)
    ax.plot(8.0, 0.5, marker="s", color=AMBER_DEEP, markersize=6.5, zorder=5)
    ax.vlines(8.0, 0, 0.5, color=AMBER_DEEP, linestyle=":", lw=1.2, zorder=3)
    ax.hlines(0.5, 0, 8.0, color=AMBER_DEEP, linestyle=":", lw=1.2, zorder=3)
    ax.text(9.2, 0.48, r"$s = \sigma_0 = 8\mathrm{px}\quad (\gamma = 0.5)$", fontsize=7.8, color=AMBER_DEEP, weight="bold", zorder=6)

    # Limit 2: s -> infty (Macro limit)
    ax.plot(36.0, 36**2 / (36**2 + 64), marker="^", color=NAVY_DEEP, markersize=6.0, zorder=5)
    ax.text(19.0, 0.90, r"$\lim_{s \to \infty} \gamma(s) = 1 \Rightarrow \mathcal{S}_{\mathrm{H-WIoU}} \to \mathrm{IoU}$",
            fontsize=7.8, color=NAVY_DEEP, weight="bold", zorder=6)

    # Formula Box in center
    formula_text = (
        r"$\mathbf{Continuous\ Homotopy\ Metric\ Space:}$" + "\n" +
        r"$\mathcal{S}_{\mathrm{H-WIoU}}(A, B) = \gamma(s)\,\mathrm{IoU}(A,B) + (1-\gamma(s))\,\exp(-\mathcal{D}_{\mathcal{W}}^2(A,B))$" + "\n" +
        r"$\bullet\ \text{Continuous}\ \mathcal{C}^\infty\ \text{deformation preserving topological gradient everywhere.}$"
    )
    ax.text(20.0, 0.28, formula_text, ha="center", va="center", fontsize=7.4, color=SLATE_DARK,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec=AMBER_DEEP, lw=1.2), zorder=5)

    ax.set_xlim(0, 40)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"$\mathrm{Spatial\ Scale\ Parameter}\ s(B) = \sqrt{w_b \cdot h_b}\ (\mathrm{pixels})$", fontsize=8.2, color=SLATE_DARK, labelpad=4)
    ax.set_ylabel(r"$\mathrm{Homotopy\ Interpolation\ Weight}\ \gamma(s)$", fontsize=8.2, color=SLATE_DARK, labelpad=4)
    ax.set_xticks([0, 8, 16, 24, 32, 40])
    ax.set_xticklabels(["0", r"$\sigma_0\,(8\mathrm{px})$", "16", "24", "32", "40"], fontsize=7.5, color=SLATE_DARK)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=7.5, color=SLATE_DARK)
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0", zorder=0)

    legend = ax.legend(loc="lower right", frameon=True, facecolor="#FFFFFF", edgecolor=BORDER_GRAY, fontsize=7.0)
    legend.get_frame().set_boxstyle("round,pad=0.3")

    plt.tight_layout()
    pdf_out = OUT_DIR / "fig2_multimetric_radar.pdf"
    png_out = OUT_DIR / "fig2_multimetric_radar.png"
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Figure 2 (Math Intuition) to {pdf_out} and {png_out}")


# ==============================================================================
# FIGURE 5: THE ARCHITECTURE PIPELINE WITH BACKPROP GRADIENT
# ==============================================================================
def render_figure5_pipeline():
    print("Generating Figure 5 (Architecture Pipeline with Backpropagation)...")
    fig, ax = plt.subplots(figsize=(11.8, 5.2), dpi=300)
    ax.set_xlim(0, 118)
    ax.set_ylim(0, 52)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Banner Header
    banner = FancyBboxPatch((2, 46.5), 114, 4.5, boxstyle="round,pad=0.2", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2, zorder=1)
    ax.add_patch(banner)
    ax.text(59, 48.7, "H-WIoU: End-to-End Homotopy Detection Architecture & Non-Vanishing Backpropagation",
            ha="center", va="center", fontsize=9.2, color=SLATE_DARK, weight="bold", zorder=2)

    # --------------------------------------------------------------------------
    # CONTAINER 1: 2D Gaussian Embedding & FPN Backbone
    # --------------------------------------------------------------------------
    c1 = FancyBboxPatch((2, 2), 27, 43, boxstyle="round,pad=0.3", facecolor="#F8FAFC", edgecolor=BORDER_GRAY, lw=1.4, zorder=1)
    ax.add_patch(c1)
    h1 = FancyBboxPatch((3.5, 40.5), 24, 3.5, boxstyle="round,pad=0.2", facecolor="#1E293B", edgecolor="none", zorder=2)
    ax.add_patch(h1)
    ax.text(15.5, 42.2, "Stage 1: Gaussian Embedding", ha="center", va="center", fontsize=7.8, color="#FFFFFF", weight="bold", zorder=3)

    # Input BBox in R^4
    b_in = FancyBboxPatch((4.5, 31.0), 22, 7.5, boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=NAVY_DEEP, lw=1.2, zorder=2)
    ax.add_patch(b_in)
    ax.text(15.5, 35.8, r"Bounding Boxes $A, B \in \mathbb{R}^4$", ha="center", va="center", fontsize=7.4, color=NAVY_DEEP, weight="bold", zorder=3)
    ax.text(15.5, 33.0, r"$A = (x_a, y_a, w_a, h_a)$", ha="center", va="center", fontsize=7.0, color=SLATE_MUTED, zorder=3)

    # Gaussian Mapping
    b_gauss = FancyBboxPatch((4.5, 20.5), 22, 8.5, boxstyle="round,pad=0.2", facecolor="#EFF6FF", edgecolor="#3B82F6", lw=1.2, zorder=2)
    ax.add_patch(b_gauss)
    ax.text(15.5, 26.2, "2D Gaussian Injection", ha="center", va="center", fontsize=7.4, color="#1E3A8A", weight="bold", zorder=3)
    ax.text(15.5, 23.2, r"$\mu = (x, y), \ \Sigma = \mathrm{diag}(\frac{w^2}{4}, \frac{h^2}{4})$", ha="center", va="center", fontsize=6.8, color="#1D4ED8", zorder=3)

    # FPN Multi-Scale Levels
    fpn_box = FancyBboxPatch((4.5, 5.0), 22, 13.5, boxstyle="round,pad=0.2", facecolor="#DBEAFE", edgecolor="#2563EB", lw=1.0, zorder=2)
    ax.add_patch(fpn_box)
    ax.text(15.5, 16.0, "ResNet-50 + FPN Pyramid", ha="center", va="center", fontsize=7.2, color="#1E40AF", weight="bold", zorder=3)
    ax.text(15.5, 12.5, r"$P_2\ (4\times) \ \dots \ P_5\ (32\times)$", ha="center", va="center", fontsize=7.0, color="#1E3A8A", zorder=3)
    ax.text(15.5, 8.5, "Lateral 1x1 + 2x Top-Down", ha="center", va="center", fontsize=6.5, color="#1D4ED8", style="italic", zorder=3)

    # Connector 1 -> 2
    ax.annotate("", xy=(33.0, 23.5), xytext=(29.5, 23.5),
                arrowprops=dict(arrowstyle="-|>", color=NAVY_DEEP, lw=2.0), zorder=5)

    # --------------------------------------------------------------------------
    # CONTAINER 2: Stage 1 RPN Homotopy Similarity Matrix S_ij
    # --------------------------------------------------------------------------
    c2 = FancyBboxPatch((34, 2), 43, 43, boxstyle="round,pad=0.3", facecolor="#FFF7ED", edgecolor=AMBER_DEEP, lw=1.8, zorder=1)
    ax.add_patch(c2)
    h2 = FancyBboxPatch((35.5, 40.5), 40, 3.5, boxstyle="round,pad=0.2", facecolor=AMBER_DEEP, edgecolor="none", zorder=2)
    ax.add_patch(h2)
    ax.text(55.5, 42.2, r"Stage 1: Homotopy Similarity Matrix $\mathbf{S}_{i,j}$ (RPN)", ha="center", va="center", fontsize=7.8, color="#FFFFFF", weight="bold", zorder=3)

    # Homotopy Similarity Formula
    b_mat = FancyBboxPatch((36.5, 28.0), 38, 10.5, boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#FB923C", lw=1.2, zorder=2)
    ax.add_patch(b_mat)
    ax.text(55.5, 35.5, r"Pairwise Homotopy Matrix: $\mathbf{S}_{i,j} = \mathcal{S}_{\mathrm{H-WIoU}}(A_i, G_j)$",
            ha="center", va="center", fontsize=7.6, color="#9A3412", weight="bold", zorder=3)
    ax.text(55.5, 31.0, r"$\mathbf{S}_{i,j} = \gamma(s_j)\,\mathrm{IoU}(A_i, G_j) + (1 - \gamma(s_j))\,\exp(-\mathcal{D}_{\mathcal{W}}^2(A_i, G_j))$",
            ha="center", va="center", fontsize=7.2, color="#C2410C", zorder=3)

    # Soft Label Assigner
    b_rpn = FancyBboxPatch((36.5, 14.5), 38, 11.5, boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#EA580C", lw=1.2, zorder=2)
    ax.add_patch(b_rpn)
    ax.text(55.5, 22.8, "Top-k Soft Positive Assigner (HLA)", ha="center", va="center", fontsize=7.6, color="#9A3412", weight="bold", zorder=3)
    ax.text(55.5, 18.5, r"Positive Candidates: $\mathrm{Top\text{-}}k\ \mathrm{on}\ \mathbf{S}_{i,j} \Rightarrow \mathrm{Survival:\ 0.94}$" + "\n" +
            "Smooth dynamic gradient support across all anchor scales",
            ha="center", va="center", fontsize=6.8, color="#7C2D12", zorder=3)

    # Proposals Output
    b_prop = FancyBboxPatch((36.5, 4.5), 38, 7.5, boxstyle="round,pad=0.2", facecolor="#FED7AA", edgecolor=AMBER_DEEP, lw=1.0, zorder=2)
    ax.add_patch(b_prop)
    ax.text(55.5, 8.2, "High-Recall Tiny Proposals (Top 2000 RoIs)", ha="center", va="center", fontsize=7.4, color="#7C2D12", weight="bold", zorder=3)

    # Connector 2 -> 3
    ax.annotate("", xy=(81.0, 23.5), xytext=(77.5, 23.5),
                arrowprops=dict(arrowstyle="-|>", color=TEAL_DEEP, lw=2.0), zorder=5)

    # --------------------------------------------------------------------------
    # CONTAINER 3: Stage 2 RoI Head & Bounded Loss with Backprop
    # --------------------------------------------------------------------------
    c3 = FancyBboxPatch((82, 2), 34, 43, boxstyle="round,pad=0.3", facecolor="#F0FDF4", edgecolor=TEAL_DEEP, lw=1.5, zorder=1)
    ax.add_patch(c3)
    h3 = FancyBboxPatch((83.5, 40.5), 31, 3.5, boxstyle="round,pad=0.2", facecolor=TEAL_DEEP, edgecolor="none", zorder=2)
    ax.add_patch(h3)
    ax.text(99.0, 42.2, "Stage 2: RoI Head & Bounded Loss", ha="center", va="center", fontsize=7.8, color="#FFFFFF", weight="bold", zorder=3)

    # RoIAlign 7x7
    b_roi = FancyBboxPatch((84.0, 31.5), 30, 7.0, boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor="#4ADE80", lw=1.2, zorder=2)
    ax.add_patch(b_roi)
    ax.text(99.0, 36.0, r"$\mathbf{7\times 7\ RoIAlign\ +\ 2\times FC(1024)}$", ha="center", va="center", fontsize=7.4, color="#14532D", weight="bold", zorder=3)
    ax.text(99.0, 33.2, "Bilinear Feature Sampling", ha="center", va="center", fontsize=6.8, color="#15803D", zorder=3)

    # Bounded Loss Box with Note
    b_loss = FancyBboxPatch((84.0, 15.5), 30, 14.0, boxstyle="round,pad=0.2", facecolor="#DCFCE7", edgecolor=TEAL_DEEP, lw=1.5, zorder=2)
    ax.add_patch(b_loss)
    ax.text(99.0, 26.5, "Bounded Regression Loss:", ha="center", va="center", fontsize=7.5, color="#14532D", weight="bold", zorder=3)
    ax.text(99.0, 22.8, r"$\mathcal{L}_{\mathrm{H-WIoU}} = 1 - \mathcal{S}_{\mathrm{H-WIoU}}$", ha="center", va="center", fontsize=8.0, color="#166534", weight="bold", zorder=3)
    ax.text(99.0, 18.5, r"$\mathbf{Strictly\ Bounded:\ } \mathcal{L} \in [0, 1]$" + "\n" + "Zero loss explosion on sub-pixel offsets",
            ha="center", va="center", fontsize=6.8, color="#14532D", style="italic", zorder=3)

    # Final Output Calibrated
    b_out = FancyBboxPatch((84.0, 4.5), 30, 8.5, boxstyle="round,pad=0.2", facecolor=TEAL_DEEP, edgecolor="none", zorder=2)
    ax.add_patch(b_out)
    ax.text(99.0, 9.8, "Final Calibrated Detections", ha="center", va="center", fontsize=7.6, color="#FFFFFF", weight="bold", zorder=3)
    ax.text(99.0, 6.8, r"$\mathrm{AI-TOD-v2\ mAP}_{50}:\ \mathbf{46.2\%}\ (+19.9\%)$", ha="center", va="center", fontsize=7.2, color="#BBF7D0", zorder=3)

    # --------------------------------------------------------------------------
    # BACKPROPAGATION GRADIENT FLOW (RED DASHED ARROW WITH NON-VANISHING FORMULA)
    # --------------------------------------------------------------------------
    bp_arrow = FancyArrowPatch((99.0, 15.0), (55.5, 4.5),
                               connectionstyle="arc3,rad=0.35",
                               arrowstyle="-|>,head_length=6,head_width=3.5",
                               color="#DC2626", lw=2.0, ls="--", zorder=6)
    ax.add_patch(bp_arrow)
    
    # Backprop formula callout
    bp_box = dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec="#DC2626", lw=1.2)
    ax.text(78.0, 9.5,
            r"$\mathbf{Backpropagation\ Gradient:}$" + "\n" +
            r"$\|\nabla_\theta\,\mathcal{L}_{\mathrm{H-WIoU}}\| = \mathcal{O}(1) > 0\quad (\mathrm{even\ if\ IoU}=0)$" + "\n" +
            r"$\nabla \propto \frac{\Delta x}{\bar{w}_{ab}^2}\quad (\mathrm{Smooth\ Linear\ Restoration})$",
            ha="center", va="center", fontsize=7.2, color="#991B1B", bbox=bp_box, zorder=7)

    pdf_out = OUT_DIR / "fig5_pipeline_architecture.pdf"
    png_out = OUT_DIR / "fig5_pipeline_architecture.png"
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Figure 5 (Architecture Pipeline with Backprop) to {pdf_out} and {png_out}")


def main():
    print("=" * 80)
    print("   GENERATING ACADEMIC MASTERPIECE JOURNAL FIGURES (IEEE TPAMI)")
    print("=" * 80)
    render_figure1_teaser()
    render_figure2_math_intuition()
    render_figure5_pipeline()
    print("=" * 80)

if __name__ == "__main__":
    main()
