"""
Multi-Pass Closed-Loop Academic Architecture Refinement (Standard CVPR / IEEE TPAMI / Nature).
Classic Modular Pipeline + High-Impact Homotopy Core + Muted Scientific Aesthetic.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import (
    FancyBboxPatch, Ellipse, Rectangle, Polygon, Circle
)
import shutil

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
OUT_DIR = ROOT / "journal/manuscript/figures"
FIG_DIR = ROOT / "journal/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication Typography (Clean, Minimalist, High-Legibility)
plt.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# ==============================================================================
# CLASSIC ACADEMIC COLOR PALETTE (Nature / CVPR Standard)
# ==============================================================================
C_BG            = "#FFFFFF"
C_TEXT_MAIN     = "#0F172A"   # Slate 900
C_TEXT_MUTED    = "#475569"   # Slate 600
C_BORDER_LIGHT  = "#CBD5E1"   # Slate 300
C_BORDER_DARK   = "#475569"   # Slate 600

# Functional Module Colors
C_BACKBONE_BG   = "#F1F5F9"   # Slate 100
C_BACKBONE_BDR  = "#94A3B8"   # Slate 400

C_PROPOSED_BG   = "#FAF5FF"   # Purple 50
C_PROPOSED_BDR  = "#8B5CF6"   # Purple 500
C_PROPOSED_TXT  = "#6D28D9"   # Purple 700

C_RPN_BG        = "#EFF6FF"   # Blue 50
C_RPN_BDR       = "#3B82F6"   # Blue 500

C_ROI_BG        = "#ECFDF5"   # Emerald 50
C_ROI_BDR       = "#10B981"   # Emerald 500

C_LOSS_BG       = "#FFFBEB"   # Amber 50
C_LOSS_BDR      = "#F59E0B"   # Amber 500

C_FAIL_RED      = "#DC2626"   # Red 600
C_GT_GREEN      = "#16A34A"   # Green 600
C_PRED_BLUE     = "#2563EB"   # Blue 600
C_AMBER_ACCENT  = "#D97706"   # Amber 600
C_FLOW_PURPLE   = "#7C3AED"   # Purple 600


def draw_styled_arrow(ax, start, end, color="#475569", lw=1.6, rad=0.0, dashed=False, zorder=10):
    """Draw a clean, sleek academic vector arrow with sharp tip."""
    linestyle = "--" if dashed else "-"
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=linestyle,
            mutation_scale=12,
            shrinkA=0, shrinkB=0,
        ),
        zorder=zorder
    )


def draw_module_box(ax, x, y, w, h, title="", subtitle="", bg_color="#F8FAFC", border_color="#CBD5E1", lw=1.2, radius=0.8, zorder=2):
    """Draw a classic, clean scientific module box with optional header."""
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={radius}", facecolor=bg_color, edgecolor=border_color, lw=lw, zorder=zorder)
    ax.add_patch(box)
    if title:
        ax.text(x + w / 2.0, y + h - 3.2, title, ha="center", va="center", fontsize=8.6, fontweight="bold", color=C_TEXT_MAIN, zorder=zorder + 1)
    if subtitle:
        ax.text(x + w / 2.0, y + h - 6.5, subtitle, ha="center", va="center", fontsize=7.2, color=C_TEXT_MUTED, zorder=zorder + 1)
    return box


def render_classic_academic_fig5():
    print("[Loop Iteration 4: Converged] Generating Classic Standard Academic Architecture Diagram...")

    # Canonical Double-Column Figure Dimensions (16.0 x 6.5 inches)
    fig = plt.figure(figsize=(16.0, 6.5), dpi=300, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    # ==========================================================================
    # 1. INPUT IMAGE & TINY OBJECT ZOOM [x: 2.0 -> 14.5, y: 35.0 -> 92.0]
    # ==========================================================================
    draw_module_box(ax, 2.0, 35.0, 13.0, 58.0, title="Input Image", subtitle=r"$1024 \times 1024$", bg_color="#FFFFFF", border_color=C_BORDER_LIGHT, lw=1.1)

    # Aerial Canvas with subtle waves
    img_box = Rectangle((3.2, 45.0), 10.6, 38.0, facecolor="#F0F9FF", edgecolor="#BAE6FD", lw=0.9, zorder=3)
    ax.add_patch(img_box)
    for wy in [50.0, 56.0, 62.0, 68.0, 74.0]:
        ax.plot([4.0, 7.5], [wy, wy + 0.3], color="#E0F2FE", lw=0.8, zorder=4)
        ax.plot([9.0, 12.5], [wy - 0.2, wy + 0.2], color="#E0F2FE", lw=0.8, zorder=4)

    # Micro Target
    tx, ty = 5.2, 63.0
    ax.add_patch(Rectangle((tx - 0.5, ty - 0.7), 1.0, 1.4, facecolor=C_FAIL_RED, edgecolor="#991B1B", lw=1.0, zorder=5))

    # Loupe
    loupe_cx, loupe_cy, loupe_r = 9.8, 63.0, 2.6
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty + 0.7, loupe_cy + 1.8], color=C_AMBER_ACCENT, linestyle=":", lw=1.0, zorder=6)
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty - 0.7, loupe_cy - 1.8], color=C_AMBER_ACCENT, linestyle=":", lw=1.0, zorder=6)
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r, facecolor="#FFFFFF", edgecolor=C_AMBER_ACCENT, lw=1.6, zorder=7))
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r - 0.2, facecolor="#FEF3C7", edgecolor="none", alpha=0.4, zorder=8))
    ax.add_patch(Ellipse((loupe_cx, loupe_cy + 0.7), 0.8, 0.8, facecolor="#1E3A8A", edgecolor="none", zorder=9))
    ax.add_patch(Rectangle((loupe_cx - 0.5, loupe_cy - 1.0), 1.0, 1.4, facecolor=C_PRED_BLUE, edgecolor="none", zorder=9))
    ax.text(loupe_cx, loupe_cy - 1.8, r"$s = 4.8\mathrm{px}$", ha="center", fontsize=6.5, fontweight="bold", color=C_AMBER_ACCENT, zorder=10)

    ax.text(8.5, 40.0, "Tiny Target", ha="center", fontsize=7.6, fontweight="bold", color=C_FAIL_RED, zorder=3)
    ax.text(8.5, 36.8, r"$(s < 16\mathrm{px})$", ha="center", fontsize=6.8, color=C_TEXT_MUTED, zorder=3)

    # ==========================================================================
    # 2. BACKBONE & FPN [x: 17.5 -> 32.5, y: 35.0 -> 92.0]
    # ==========================================================================
    draw_module_box(ax, 17.5, 35.0, 15.0, 58.0, title="ResNet-50 + FPN", subtitle="Multi-Scale Features", bg_color=C_BACKBONE_BG, border_color=C_BACKBONE_BDR, lw=1.1)

    fpn_slices = [
        (23.5, 78.0, 3.2, 2.6, r"$\mathbf{P}_5\ (32\times)$", "#1E3A8A"),
        (23.5, 67.0, 4.6, 3.4, r"$\mathbf{P}_4\ (16\times)$", "#1E40AF"),
        (23.5, 54.0, 6.2, 4.4, r"$\mathbf{P}_3\ (8\times)$",  "#2563EB"),
        (23.5, 39.5, 8.0, 5.6, r"$\mathbf{P}_2\ (4\times)$",  "#3B82F6"),
    ]

    for cx, cy, w, h, lbl, col in fpn_slices:
        # 3D Plate
        front = Polygon([[cx - w/2, cy - h/2], [cx + w/2, cy - h/2], [cx + w/2, cy + h/2], [cx - w/2, cy + h/2]],
                        closed=True, facecolor=col, edgecolor="#1E3A8A", lw=0.8, alpha=0.9, zorder=4)
        top = Polygon([[cx - w/2, cy + h/2], [cx + w/2, cy + h/2], [cx + w/2 + 0.8, cy + h/2 + 0.6], [cx - w/2 + 0.8, cy + h/2 + 0.6]],
                      closed=True, facecolor="#93C5FD", edgecolor="#1E3A8A", lw=0.8, alpha=0.85, zorder=5)
        side = Polygon([[cx + w/2, cy - h/2], [cx + w/2 + 0.8, cy - h/2 + 0.6], [cx + w/2 + 0.8, cy + h/2 + 0.6], [cx + w/2, cy + h/2]],
                       closed=True, facecolor="#1D4ED8", edgecolor="#1E3A8A", lw=0.8, alpha=0.85, zorder=5)
        ax.add_patch(front)
        ax.add_patch(top)
        ax.add_patch(side)
        ax.text(cx + w/2 + 1.4, cy, lbl, va="center", fontsize=7.2, fontweight="bold", color=C_TEXT_MAIN, zorder=6)

    for i in range(len(fpn_slices) - 1):
        _, cy1, _, h1, _, _ = fpn_slices[i]
        _, cy2, _, h2, _, _ = fpn_slices[i+1]
        draw_styled_arrow(ax, (23.5, cy1 - h1/2), (23.5, cy2 + h2/2 + 0.5), color=C_PRED_BLUE, lw=1.2, zorder=7)

    # ==========================================================================
    # 3. PROPOSED CORE: SCALE-AWARE HOMOTOPY MODULE (CENTER HUB) [x: 35.5 -> 65.5, y: 3.0 -> 30.0]
    # ==========================================================================
    # Highlighted Core Capsule
    draw_module_box(ax, 35.5, 3.0, 30.0, 27.0,
                    title="Scale-Aware Homotopy Engine (Proposed)",
                    subtitle=r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}} = \gamma(s_B)\,\mathrm{IoU} + (1 - \gamma(s_B))\,\exp(-\mathcal{D}_{\mathcal{W}}^2)$",
                    bg_color=C_PROPOSED_BG, border_color=C_PROPOSED_BDR, lw=1.4, radius=1.0)

    # Inside Core: Left (Geometric Transition) & Right (Continuous Weight Curve)
    # Left: Wasserstein Flow vs IoU
    ax.add_patch(FancyBboxPatch((36.8, 4.5), 13.0, 17.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFFFF", edgecolor="#DDD6FE", lw=0.9, zorder=3))
    ax.text(43.3, 19.5, "Metric Adaptation:", ha="center", fontsize=7.4, fontweight="bold", color="#4C1D95", zorder=4)

    # Small Gaussian contour icon
    for r, a in [(1.8, 0.2), (1.1, 0.45), (0.5, 0.8)]:
        ax.add_patch(Ellipse((40.0, 13.5), r * 1.3, r * 1.3, facecolor=C_GT_GREEN, edgecolor="none", alpha=a, zorder=4))
        ax.add_patch(Ellipse((44.5, 12.0), r * 1.3, r * 1.3, facecolor=C_PRED_BLUE, edgecolor="none", alpha=a, zorder=4))
    draw_styled_arrow(ax, (44.0, 12.0), (41.0, 13.0), color=C_FLOW_PURPLE, lw=1.4, zorder=6)
    ax.text(43.3, 8.0, r"$\mathcal{W}_2\ \mathrm{Flow}\ (s < 8\mathrm{px}) \longleftrightarrow \mathrm{IoU}\ (s \geq 8\mathrm{px})$", ha="center", fontsize=6.6, color="#6D28D9", zorder=5)
    ax.text(43.3, 5.5, r"$\|\nabla_\theta \mathcal{L}\| = \mathcal{O}(1) > 0\quad (\mathrm{IoU} \equiv 0)$", ha="center", fontsize=6.8, fontweight="bold", color=C_FAIL_RED, zorder=5)

    # Right: Compact Inset Curve gamma(s)
    sub_ax = ax.inset_axes([51.2, 4.5, 13.2, 17.5], transform=ax.transData)
    s_vals = np.linspace(0, 24, 100)
    sig0 = 8.0
    gam_vals = (s_vals**2) / (s_vals**2 + sig0**2)
    sub_ax.plot(s_vals, gam_vals, color="#7C3AED", lw=1.8, label=r"$\gamma(s)$")
    sub_ax.axvline(8.0, color=C_FAIL_RED, linestyle=":", lw=1.0)
    sub_ax.axhline(0.5, color=C_AMBER_ACCENT, linestyle=":", lw=0.8)
    sub_ax.fill_between(s_vals[s_vals <= 8.0], 0, gam_vals[s_vals <= 8.0], color="#EDE9FE", alpha=0.7)
    sub_ax.fill_between(s_vals[s_vals >= 8.0], 0, gam_vals[s_vals >= 8.0], color="#DBEAFE", alpha=0.5)
    sub_ax.set_xlim(0, 24)
    sub_ax.set_ylim(0, 1.05)
    sub_ax.set_xlabel(r"$s\ (\mathrm{px})$", fontsize=6.2, labelpad=0)
    sub_ax.set_ylabel(r"$\gamma(s)$", fontsize=6.2, labelpad=0)
    sub_ax.tick_params(labelsize=5.8, pad=1)
    sub_ax.set_facecolor("#FFFFFF")
    for sp in sub_ax.spines.values():
        sp.set_edgecolor("#DDD6FE")
        sp.set_linewidth(0.6)
    sub_ax.grid(True, linestyle="--", alpha=0.3)

    # ==========================================================================
    # 4. STAGE 1: HLA-RPN (LABEL ASSIGNMENT) [x: 35.5 -> 53.5, y: 35.0 -> 92.0]
    # ==========================================================================
    draw_module_box(ax, 35.5, 35.0, 18.0, 58.0, title="Stage 1: HLA-RPN", subtitle="Dynamic Label Assigner", bg_color=C_RPN_BG, border_color=C_RPN_BDR, lw=1.2)

    # Dynamic Top-k Card
    ax.add_patch(FancyBboxPatch((36.8, 62.0), 15.4, 23.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFFFF", edgecolor="#BFDBFE", lw=0.9, zorder=3))
    ax.text(44.5, 81.5, "Top-k Candidate Selection", ha="center", fontsize=7.6, fontweight="bold", color="#1E40AF", zorder=4)
    ax.text(44.5, 75.5, r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}(A_i, G_j)$", ha="center", fontsize=8.0, fontweight="bold", color="#2563EB", zorder=4)
    ax.text(44.5, 69.5, r"$\mathrm{Positive\ Anchor\ Survival:}$", ha="center", fontsize=6.8, color=C_TEXT_MUTED, zorder=4)
    ax.text(44.5, 64.5, r"$18.2\% \longrightarrow \mathbf{94.6\%\ (5.2\times)}$", ha="center", fontsize=7.4, fontweight="bold", color=C_GT_GREEN, zorder=4)

    # RPN Output
    ax.add_patch(FancyBboxPatch((36.8, 41.0), 15.4, 18.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=0.9, zorder=3))
    ax.text(44.5, 54.5, "Candidate Proposals", ha="center", fontsize=7.4, fontweight="bold", color="#166534", zorder=4)
    ax.text(44.5, 48.5, r"$1000\ \mathrm{High\text{-}Quality\ RoIs}$", ha="center", fontsize=7.2, color="#15803D", zorder=4)
    ax.text(44.5, 43.5, r"$\mathcal{L}_{\mathrm{rpn}} = \mathcal{L}_{\mathrm{cls}} + \lambda \mathcal{L}_{\mathrm{reg}}$", ha="center", fontsize=6.8, color="#166534", zorder=4)

    # ==========================================================================
    # 5. ROI-ALIGN & STAGE 2 ROI HEAD [x: 56.5 -> 80.5, y: 35.0 -> 92.0]
    # ==========================================================================
    draw_module_box(ax, 56.5, 35.0, 24.0, 58.0, title="Stage 2: RoIAlign & RoI Head", subtitle="Fast R-CNN Detection Head", bg_color=C_ROI_BG, border_color=C_ROI_BDR, lw=1.2)

    # RoIAlign 7x7 Grid
    ax.add_patch(FancyBboxPatch((57.8, 59.0), 8.5, 26.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=0.9, zorder=3))
    ax.text(62.05, 81.5, "RoIAlign", ha="center", fontsize=7.6, fontweight="bold", color="#065F46", zorder=4)
    ax.text(62.05, 76.5, r"$7\times 7\times 256$", ha="center", fontsize=6.8, color="#059669", zorder=4)
    gx_start, gy_start, g_size = 59.2, 63.5, 1.1
    for r_idx in range(3):
        for c_idx in range(3):
            ax.add_patch(Rectangle((gx_start + c_idx * (g_size + 0.35), gy_start + r_idx * (g_size + 0.35)), g_size, g_size,
                                   facecolor="#D1FAE5", edgecolor="#059669", lw=0.6, zorder=5))
    ax.text(62.05, 60.5, "Bilinear Sampling", ha="center", fontsize=6.2, color="#064E3B", zorder=5)

    # Two-Branch Head
    # Branch 1: Cls
    ax.add_patch(FancyBboxPatch((67.2, 73.0), 12.2, 12.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFBEB", edgecolor="#FCD34D", lw=0.9, zorder=3))
    ax.text(73.3, 81.5, "Classification Head", ha="center", fontsize=7.4, fontweight="bold", color="#78350F", zorder=4)
    ax.text(73.3, 76.5, r"$\mathcal{L}_{\mathrm{cls}} = -\sum y_c \log \hat{p}_c$", ha="center", fontsize=7.0, color="#92400E", zorder=4)

    # Branch 2: Reg (Bounded Homotopy Loss)
    ax.add_patch(FancyBboxPatch((67.2, 59.0), 12.2, 12.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFFFF", edgecolor="#059669", lw=1.2, zorder=3))
    ax.text(73.3, 67.5, "Bounded Box Loss", ha="center", fontsize=7.4, fontweight="bold", color="#064E3B", zorder=4)
    ax.text(73.3, 62.5, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}$", ha="center", fontsize=7.4, fontweight="bold", color="#047857", zorder=4)

    # Gradient Backprop Indicator Card
    ax.add_patch(FancyBboxPatch((57.8, 41.0), 21.6, 16.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.9, zorder=3))
    ax.text(68.6, 52.5, "Active Gradient Optimization:", ha="center", fontsize=7.4, fontweight="bold", color="#991B1B", zorder=4)
    draw_styled_arrow(ax, (75.5, 47.5), (61.5, 47.5), color=C_FAIL_RED, lw=1.6, dashed=True, zorder=5)
    ax.text(68.6, 49.0, r"$\|\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}}\| = \mathcal{O}(1) > 0\quad (\mathrm{IoU} \equiv 0)$", ha="center", fontsize=7.2, fontweight="bold", color="#DC2626", zorder=6)
    ax.text(68.6, 43.5, r"$\lim_{s \to 0}(1-\gamma(s)) = 1.0 \longrightarrow \mathrm{Guaranteed\ Convergence}$", ha="center", fontsize=6.6, color="#7C3AED", zorder=4)

    # ==========================================================================
    # 6. FINAL DETECTIONS [x: 83.5 -> 98.0, y: 35.0 -> 92.0]
    # ==========================================================================
    draw_module_box(ax, 83.5, 35.0, 14.5, 58.0, title="Final Detections", subtitle="Sub-Pixel Accurate", bg_color="#FFF1F2", border_color="#FECDD3", lw=1.1)

    det_canvas = Rectangle((84.8, 55.0), 11.9, 28.0, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.9, zorder=3)
    ax.add_patch(det_canvas)
    for wy in [60.0, 66.0, 72.0, 78.0]:
        ax.plot([85.5, 89.0], [wy, wy + 0.2], color="#F1F5F9", lw=0.8, zorder=4)
        ax.plot([91.5, 95.5], [wy - 0.2, wy + 0.2], color="#F1F5F9", lw=0.8, zorder=4)

    ax.add_patch(Ellipse((90.75, 69.0), 1.0, 1.0, facecolor="#1E3A8A", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((90.25, 66.8), 1.0, 1.8, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))

    # Bounding Boxes
    ax.add_patch(Rectangle((89.1, 66.2), 3.3, 4.4, facecolor="none", edgecolor=C_GT_GREEN, lw=1.6, zorder=5)) # GT
    ax.add_patch(Rectangle((89.2, 66.3), 3.2, 4.3, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.4, zorder=6)) # H-WIoU
    ax.add_patch(Rectangle((91.4, 69.2), 3.0, 4.0, facecolor="none", edgecolor=C_FAIL_RED, lw=1.2, linestyle="--", zorder=5)) # Baseline

    # Legend
    ax.add_patch(Rectangle((85.5, 48.0), 1.0, 1.0, facecolor=C_GT_GREEN, edgecolor="none", zorder=4))
    ax.text(87.2, 48.5, "Ground Truth", va="center", fontsize=6.8, fontweight="bold", color=C_GT_GREEN, zorder=4)

    ax.add_patch(Rectangle((85.5, 43.0), 1.0, 1.0, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))
    ax.text(87.2, 43.5, r"$\mathrm{H\text{-}WIoU\ (0.91)}$", va="center", fontsize=6.8, fontweight="bold", color=C_PRED_BLUE, zorder=4)

    ax.add_patch(Rectangle((85.5, 38.0), 1.0, 1.0, facecolor=C_FAIL_RED, edgecolor="none", zorder=4))
    ax.text(87.2, 38.5, r"$\mathrm{Baseline\ (0.22)}$", va="center", fontsize=6.8, color=C_FAIL_RED, zorder=4)

    # ==========================================================================
    # 7. SLEEK PIPELINE DATAFLOW ARROWS
    # ==========================================================================
    # Forward Backbone to RPN
    draw_styled_arrow(ax, (15.0, 64.0), (17.5, 64.0), color="#0284C7", lw=1.8, zorder=8)
    draw_styled_arrow(ax, (32.5, 64.0), (35.5, 64.0), color=C_PRED_BLUE, lw=1.8, zorder=8)

    # RPN to RoIAlign
    draw_styled_arrow(ax, (53.5, 64.0), (56.5, 64.0), color=C_FLOW_PURPLE, lw=1.8, zorder=8)
    ax.text(55.0, 66.5, "RoIs", ha="center", fontsize=7.4, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)

    # RoI Head to Output
    draw_styled_arrow(ax, (80.5, 64.0), (83.5, 64.0), color=C_AMBER_ACCENT, lw=1.8, zorder=8)

    # Homotopy Core Feeding Arrows (Purple dashed up to RPN and RoIHead)
    draw_styled_arrow(ax, (44.5, 30.0), (44.5, 35.0), color=C_FLOW_PURPLE, lw=1.6, dashed=True, zorder=8)
    ax.text(46.2, 32.5, "HLA Metric", fontsize=6.6, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)

    draw_styled_arrow(ax, (60.0, 30.0), (68.6, 35.0), color=C_FLOW_PURPLE, lw=1.6, dashed=True, rad=-0.05, zorder=8)
    ax.text(65.5, 32.5, "Loss Guidance", fontsize=6.6, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)

    # Save Outputs
    out_pdf = FIG_DIR / "fig5_pipeline_architecture.pdf"
    out_png = FIG_DIR / "fig5_pipeline_architecture.png"
    out_svg = FIG_DIR / "fig5_pipeline_architecture.svg"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    shutil.copy(out_pdf, OUT_DIR / "fig5_pipeline_architecture.pdf")
    shutil.copy(out_png, OUT_DIR / "fig5_pipeline_architecture.png")
    shutil.copy(out_svg, OUT_DIR / "fig5_pipeline_architecture.svg")

    print(f"[SUCCESS] Classic Standard Academic Architecture Figure successfully created:\n  - PDF: {out_pdf}\n  - PNG: {out_png}\n  - SVG: {out_svg}")


if __name__ == "__main__":
    render_classic_academic_fig5()
