"""
Ultra-Polished Panoramic Architecture Masterpiece (IEEE TPAMI / CVPR Standard).
Double-Column Full Width Geometry, Clean Typography, Zero Artifacts, Perfect Aspect Ratio.
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

# Publication Typography
plt.rcParams.update({
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

# ==============================================================================
# MINIMALIST SCIENTIFIC COLOR PALETTE
# ==============================================================================
C_BG            = "#FFFFFF"
C_TEXT_MAIN     = "#0F172A"   # Slate 900
C_TEXT_MUTED    = "#475569"   # Slate 600
C_BORDER_LIGHT  = "#CBD5E1"   # Slate 300

# Mechanism Accent Colors
C_GT_GREEN      = "#15803D"   # Emerald Green
C_PRED_BLUE     = "#2563EB"   # Royal Blue
C_FAIL_RED      = "#DC2626"   # Ruby Red
C_FLOW_PURPLE   = "#7C3AED"   # Violet
C_AMBER_ACCENT  = "#D97706"   # Warm Amber


def draw_styled_arrow(ax, start, end, color="#475569", lw=1.8, rad=0.0, dashed=False, zorder=10):
    """Draw a clean, sleek academic vector arrow."""
    linestyle = "--" if dashed else "-"
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=linestyle,
            mutation_scale=14,
            shrinkA=0, shrinkB=0,
        ),
        zorder=zorder
    )


def draw_isometric_feature_slice(ax, cx, cy, w, h, depth=1.6, label="", face_color="#3B82F6", alpha=0.9, zorder=4):
    """Draw an elegant semi-transparent 3D isometric feature plane."""
    dx = depth * 0.707
    dy = depth * 0.500
    ox = cx - w / 2.0
    oy = cy - h / 2.0

    front = Polygon(
        [[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]],
        closed=True, facecolor=face_color, edgecolor="#1E40AF", linewidth=0.9, alpha=alpha, zorder=zorder
    )
    ax.add_patch(front)

    top = Polygon(
        [[ox, oy + h], [ox + w, oy + h], [ox + w + dx, oy + h + dy], [ox + dx, oy + h + dy]],
        closed=True, facecolor="#93C5FD", edgecolor="#1E40AF", linewidth=0.9, alpha=alpha * 0.9, zorder=zorder + 1
    )
    ax.add_patch(top)

    side = Polygon(
        [[ox + w, oy], [ox + w + dx, oy + dy], [ox + w + dx, oy + h + dy], [ox + w, oy + h]],
        closed=True, facecolor="#1D4ED8", edgecolor="#1E40AF", linewidth=0.9, alpha=alpha * 0.9, zorder=zorder + 1
    )
    ax.add_patch(side)

    if label:
        ax.text(ox + w + dx + 0.8, oy + h / 2.0 + dy / 2.0, label, va="center", fontsize=8.0, fontweight="bold", color=C_TEXT_MAIN, zorder=zorder + 2)


def render_ultra_polished_fig5():
    print("Rendering Ultra-Polished Architecture Figure (18x6 inches, IEEE TPAMI Standard)...")

    # High-Resolution Master Canvas (18 x 5.8 inches - Perfect Proportion for Paper)
    fig = plt.figure(figsize=(18, 5.8), dpi=300, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    # ==========================================================================
    # 1. INPUT CANVAS & CIRCULAR LOUPE [x: 1.5 -> 15.5]
    # ==========================================================================
    ax.add_patch(FancyBboxPatch((1.5, 4.0), 14.0, 92.0, boxstyle="round,pad=0.0,rounding_size=1.0", facecolor="#F8FAFC", edgecolor=C_BORDER_LIGHT, lw=1.1, zorder=1))
    ax.text(8.5, 91.0, "Input Aerial Image", ha="center", fontsize=9.2, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
    ax.text(8.5, 87.0, r"$1024 \times 1024\ \mathrm{px}$", ha="center", fontsize=7.6, color=C_TEXT_MUTED, zorder=3)

    img_rect = Rectangle((2.7, 24.0), 11.6, 60.0, facecolor="#F0F9FF", edgecolor="#BAE6FD", lw=1.0, zorder=2)
    ax.add_patch(img_rect)
    for wy in [32.0, 40.0, 48.0, 56.0, 64.0, 72.0, 78.0]:
        ax.plot([3.5, 7.5], [wy, wy + 0.3], color="#E0F2FE", lw=0.8, zorder=3)
        ax.plot([9.0, 13.5], [wy - 0.2, wy + 0.2], color="#E0F2FE", lw=0.8, zorder=3)

    tx, ty = 5.5, 54.0
    ax.add_patch(Rectangle((tx - 0.6, ty - 0.8), 1.2, 1.6, facecolor=C_FAIL_RED, edgecolor="#991B1B", lw=1.2, zorder=5))

    loupe_cx, loupe_cy, loupe_r = 10.5, 54.0, 3.0
    ax.plot([tx + 0.6, loupe_cx - loupe_r], [ty + 0.8, loupe_cy + 2.0], color=C_AMBER_ACCENT, linestyle=":", lw=1.2, zorder=6)
    ax.plot([tx + 0.6, loupe_cx - loupe_r], [ty - 0.8, loupe_cy - 2.0], color=C_AMBER_ACCENT, linestyle=":", lw=1.2, zorder=6)
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r, facecolor="#FFFFFF", edgecolor=C_AMBER_ACCENT, lw=2.0, zorder=7))
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r - 0.2, facecolor="#FEF3C7", edgecolor="none", alpha=0.45, zorder=8))
    ax.add_patch(Ellipse((loupe_cx, loupe_cy + 0.9), 1.0, 1.0, facecolor="#1E3A8A", edgecolor="none", zorder=9))
    ax.add_patch(Rectangle((loupe_cx - 0.7, loupe_cy - 1.2), 1.4, 1.8, facecolor=C_PRED_BLUE, edgecolor="none", zorder=9))
    ax.text(loupe_cx, loupe_cy - 2.1, r"$s = 4.8\mathrm{px}$", ha="center", fontsize=7.2, fontweight="bold", color=C_AMBER_ACCENT, zorder=10)

    ax.text(8.5, 13.0, "Micro Swimmer Target", ha="center", fontsize=7.8, fontweight="bold", color=C_FAIL_RED, zorder=3)
    ax.text(8.5, 8.5, r"$\mathrm{Scale\ Range:\ }s \in [2, 32]\mathrm{px}$", ha="center", fontsize=7.2, color=C_TEXT_MUTED, zorder=3)

    # ==========================================================================
    # 2. RESNET-50 + FPN 3D TENSORS [x: 17.5 -> 35.0]
    # ==========================================================================
    ax.add_patch(FancyBboxPatch((17.5, 4.0), 17.5, 92.0, boxstyle="round,pad=0.0,rounding_size=1.0", facecolor="#F8FAFC", edgecolor=C_BORDER_LIGHT, lw=1.1, zorder=1))
    ax.text(26.25, 91.0, "ResNet-50 + FPN", ha="center", fontsize=9.2, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
    ax.text(26.25, 87.0, "Feature Pyramid Network", ha="center", fontsize=7.6, color=C_TEXT_MUTED, zorder=3)

    fpn_layers = [
        (22.0, 74.0, 3.4, 3.0, 1.0, r"$\mathbf{P}_5\ (32\times)$", "#1E3A8A"),
        (22.0, 59.0, 5.0, 4.0, 1.2, r"$\mathbf{P}_4\ (16\times)$", "#1E40AF"),
        (22.0, 41.5, 7.0, 5.2, 1.4, r"$\mathbf{P}_3\ (8\times)$",  "#2563EB"),
        (22.0, 20.5, 9.2, 6.6, 1.6, r"$\mathbf{P}_2\ (4\times)$",  "#3B82F6"),
    ]

    for cx, cy, w, h, dep, lbl, col in fpn_layers:
        draw_isometric_feature_slice(ax, cx, cy, w, h, depth=dep, label=lbl, face_color=col, zorder=3)

    for i in range(len(fpn_layers) - 1):
        _, cy1, _, h1, _, _, _ = fpn_layers[i]
        _, cy2, _, h2, _, _, _ = fpn_layers[i+1]
        draw_styled_arrow(ax, (22.0, cy1 - h1/2), (22.0, cy2 + h2/2 + 0.8), color=C_PRED_BLUE, lw=1.4, zorder=6)

    # Feature Output Hub
    ax.add_patch(FancyBboxPatch((28.5, 7.5), 5.5, 8.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=1.0, zorder=4))
    ax.text(31.25, 12.8, r"$\mathbf{P}_2\text{--}\mathbf{P}_5$", ha="center", fontsize=7.8, fontweight="bold", color="#1D4ED8", zorder=5)
    ax.text(31.25, 9.8, r"$256\text{-}\mathrm{d}$", ha="center", fontsize=7.0, color="#2563EB", zorder=5)

    # ==========================================================================
    # 3. STAGE 1: HLA-RPN (HOMOTOPY LABEL ASSIGNMENT) [x: 37.0 -> 60.5]
    # ==========================================================================
    ax.add_patch(FancyBboxPatch((37.0, 4.0), 23.5, 92.0, boxstyle="round,pad=0.0,rounding_size=1.0", facecolor="#FAF5FF", edgecolor="#DDD6FE", lw=1.2, zorder=1))
    ax.text(48.75, 91.0, "Stage 1: HLA-RPN", ha="center", fontsize=9.2, fontweight="bold", color="#5B21B6", zorder=3)
    ax.text(48.75, 87.0, "Scale-Homotopy Candidate Generation", ha="center", fontsize=7.6, color="#7C3AED", zorder=3)

    # Sub-Card (a): Discrete IoU vs (b) W2 Flow
    ax.add_patch(FancyBboxPatch((38.5, 48.0), 10.0, 36.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#FEF2F2", edgecolor="#FECACA", lw=1.0, zorder=2))
    ax.text(43.5, 79.5, "Discrete IoU", ha="center", fontsize=8.0, fontweight="bold", color=C_FAIL_RED, zorder=3)
    ax.add_patch(Rectangle((39.5, 64.0), 3.4, 3.4, facecolor="#DCFCE7", edgecolor=C_GT_GREEN, lw=1.4, zorder=4))
    ax.text(41.2, 65.7, "GT", ha="center", va="center", fontsize=7.0, fontweight="bold", color=C_GT_GREEN, zorder=5)
    ax.add_patch(Rectangle((44.0, 61.0), 3.4, 3.4, facecolor="none", edgecolor=C_FAIL_RED, lw=1.4, linestyle="--", zorder=4))
    ax.text(45.7, 62.7, "Anc", ha="center", va="center", fontsize=6.8, fontweight="bold", color=C_FAIL_RED, zorder=5)
    ax.plot([42.5, 43.8], [64.8, 66.1], color=C_FAIL_RED, lw=2.0, zorder=6)
    ax.plot([42.5, 43.8], [66.1, 64.8], color=C_FAIL_RED, lw=2.0, zorder=6)
    ax.text(43.5, 55.5, r"$\mathrm{IoU} = 0 \Rightarrow \nabla \equiv 0$", ha="center", fontsize=7.6, fontweight="bold", color="#991B1B", zorder=5)
    ax.text(43.5, 51.5, "Anchor Starvation", ha="center", fontsize=7.0, color="#B91C1C", zorder=5)

    ax.add_patch(FancyBboxPatch((49.5, 48.0), 10.0, 36.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=1.0, zorder=2))
    ax.text(54.5, 79.5, r"$\mathcal{W}_2\ \mathrm{Flow}$", ha="center", fontsize=8.0, fontweight="bold", color=C_GT_GREEN, zorder=3)
    for r, a in [(2.6, 0.15), (1.8, 0.35), (0.9, 0.7)]:
        ax.add_patch(Ellipse((52.0, 65.5), r * 1.3, r * 1.3, facecolor=C_GT_GREEN, edgecolor="none", alpha=a, zorder=3))
        ax.add_patch(Ellipse((57.0, 63.0), r * 1.3, r * 1.3, facecolor=C_PRED_BLUE, edgecolor="none", alpha=a, zorder=3))
    draw_styled_arrow(ax, (56.5, 63.0), (53.0, 65.0), color=C_FLOW_PURPLE, lw=1.8, zorder=6)
    ax.text(54.5, 55.5, r"$\mathcal{S}_{\mathcal{W}} > 0 \Rightarrow \nabla \neq 0$", ha="center", fontsize=7.6, fontweight="bold", color="#065F46", zorder=5)
    ax.text(54.5, 51.5, "Continuous Pull", ha="center", fontsize=7.0, color="#047857", zorder=5)

    # Dynamic Top-k Assignment Card (Bottom)
    ax.add_patch(FancyBboxPatch((38.5, 7.5), 21.0, 37.5, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#FFFFFF", edgecolor="#DDD6FE", lw=1.1, zorder=2))
    ax.text(49.0, 40.5, "Dynamic Cost Matrix Assignment:", ha="center", fontsize=7.8, fontweight="bold", color="#4C1D95", zorder=3)
    ax.text(49.0, 33.5, r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}(A_i, G_j) \in [0, 1]$", ha="center", fontsize=8.6, fontweight="bold", color="#5B21B6", zorder=3)
    ax.text(49.0, 24.5, r"$\mathrm{Top\text{-}}k\ \mathrm{Positive\ Selection}\ (18.2\% \to \mathbf{94.6\%})$", ha="center", fontsize=7.8, fontweight="bold", color=C_GT_GREEN, zorder=3)
    ax.text(49.0, 16.5, r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}} = \gamma(s)\,\mathrm{IoU} + (1-\gamma(s))\,\exp(-\mathcal{D}_{\mathcal{W}}^2)$", ha="center", fontsize=7.8, fontweight="bold", color="#92400E", zorder=3)
    ax.text(49.0, 10.5, r"$\mathrm{where}\ \gamma(s) = s^2/(s^2+\sigma_0^2)\quad (\sigma_0 = 8.0\mathrm{px})$", ha="center", fontsize=7.2, color="#B45309", zorder=3)

    # ==========================================================================
    # 4. ROI-ALIGN & STAGE 2 ROI HEAD [x: 62.5 -> 83.5]
    # ==========================================================================
    ax.add_patch(FancyBboxPatch((62.5, 4.0), 21.0, 92.0, boxstyle="round,pad=0.0,rounding_size=1.0", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=1.2, zorder=1))
    ax.text(73.0, 91.0, "Stage 2: RoIAlign & Head", ha="center", fontsize=9.2, fontweight="bold", color="#065F46", zorder=3)
    ax.text(73.0, 87.0, "High-Precision Boundary Regression", ha="center", fontsize=7.6, color="#059669", zorder=3)

    # RoIAlign Bilinear Grid (7x7)
    ax.add_patch(FancyBboxPatch((63.8, 52.0), 8.5, 32.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#FFFFFF", edgecolor="#A7F3D0", lw=1.0, zorder=2))
    ax.text(68.05, 79.5, "RoIAlign", ha="center", fontsize=8.0, fontweight="bold", color="#065F46", zorder=3)
    ax.text(68.05, 75.5, r"$7\times 7\times 256$", ha="center", fontsize=7.2, color="#059669", zorder=3)
    gx_start, gy_start, g_size = 65.2, 60.0, 1.2
    for r_idx in range(3):
        for c_idx in range(3):
            ax.add_patch(Rectangle((gx_start + c_idx * (g_size + 0.4), gy_start + r_idx * (g_size + 0.4)), g_size, g_size,
                                   facecolor="#D1FAE5", edgecolor="#059669", lw=0.7, zorder=4))
    ax.text(68.05, 55.5, "Bilinear Sampling", ha="center", fontsize=6.8, color="#064E3B", zorder=5)

    # Two-Branch Fast R-CNN Head
    ax.add_patch(FancyBboxPatch((73.2, 69.0), 9.3, 15.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#FFFBEB", edgecolor="#FCD34D", lw=1.0, zorder=2))
    ax.text(77.85, 78.5, "Classification Head", ha="center", fontsize=7.6, fontweight="bold", color="#78350F", zorder=3)
    ax.text(77.85, 73.0, r"$\mathcal{L}_{\mathrm{cls}} = \mathrm{Cross\text{-}Entropy}$", ha="center", fontsize=7.2, color="#92400E", zorder=3)

    ax.add_patch(FancyBboxPatch((73.2, 52.0), 9.3, 15.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#FFFFFF", edgecolor="#059669", lw=1.2, zorder=2))
    ax.text(77.85, 61.5, "Bounded Box Loss", ha="center", fontsize=7.6, fontweight="bold", color="#064E3B", zorder=3)
    ax.text(77.85, 56.0, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}$", ha="center", fontsize=7.6, fontweight="bold", color="#047857", zorder=3)

    # Active Gradient Backprop
    ax.add_patch(FancyBboxPatch((63.8, 7.5), 18.7, 41.5, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#FEF2F2", edgecolor="#FECACA", lw=1.1, zorder=2))
    ax.text(73.15, 43.5, "Active Gradient Optimization:", ha="center", fontsize=8.0, fontweight="bold", color="#991B1B", zorder=3)
    draw_styled_arrow(ax, (80.0, 36.5), (66.0, 36.5), color=C_FAIL_RED, lw=1.8, dashed=True, zorder=5)
    ax.text(73.15, 38.5, r"$\|\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}}\| = \mathcal{O}(1) > 0\ (\mathrm{IoU} \equiv 0)$", ha="center", fontsize=7.8, fontweight="bold", color="#DC2626", zorder=6)
    ax.text(73.15, 27.5, r"$\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}} = (1-\gamma(s_B))\,\nabla_\theta \mathcal{L}_{\mathcal{W}} \neq 0$", ha="center", fontsize=7.8, fontweight="bold", color="#991B1B", zorder=4)
    ax.text(73.15, 18.5, r"$\lim_{s \to 0}(1-\gamma(s)) = 1.0 \longrightarrow \mathrm{Pure\ W_2\ Transport}$", ha="center", fontsize=7.2, color="#7C3AED", zorder=4)
    ax.text(73.15, 11.5, "Guaranteed Convergence on Tiny Objects", ha="center", fontsize=7.2, fontweight="bold", color=C_GT_GREEN, zorder=4)

    # ==========================================================================
    # 5. FINAL DETECTION CANVAS [x: 85.5 -> 98.5]
    # ==========================================================================
    ax.add_patch(FancyBboxPatch((85.5, 4.0), 13.0, 92.0, boxstyle="round,pad=0.0,rounding_size=1.0", facecolor="#FFF1F2", edgecolor="#FECDD3", lw=1.1, zorder=1))
    ax.text(92.0, 91.0, "Final Detections", ha="center", fontsize=9.2, fontweight="bold", color="#9F1239", zorder=3)
    ax.text(92.0, 87.0, "Sub-Pixel Accurate", ha="center", fontsize=7.6, color="#BE123C", zorder=3)

    det_box = Rectangle((86.8, 38.0), 10.4, 46.0, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=1.0, zorder=2)
    ax.add_patch(det_box)
    for wy in [44.0, 52.0, 60.0, 68.0, 74.0]:
        ax.plot([87.5, 91.0], [wy, wy + 0.2], color="#F1F5F9", lw=0.8, zorder=3)
        ax.plot([93.0, 96.5], [wy - 0.2, wy + 0.2], color="#F1F5F9", lw=0.8, zorder=3)

    ax.add_patch(Ellipse((92.0, 61.0), 1.1, 1.1, facecolor="#1E3A8A", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((91.45, 58.5), 1.1, 2.0, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))

    # Bounding Boxes
    ax.add_patch(Rectangle((90.2, 57.8), 3.6, 4.8, facecolor="none", edgecolor=C_GT_GREEN, lw=1.8, zorder=5)) # GT
    ax.add_patch(Rectangle((90.3, 57.9), 3.5, 4.7, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.6, zorder=6)) # H-WIoU
    ax.add_patch(Rectangle((92.6, 61.2), 3.3, 4.4, facecolor="none", edgecolor=C_FAIL_RED, lw=1.4, linestyle="--", zorder=5)) # IoU

    # Legend
    ax.add_patch(Rectangle((87.0, 27.5), 1.2, 1.2, facecolor=C_GT_GREEN, edgecolor="none", zorder=4))
    ax.text(88.8, 28.1, "Ground Truth", va="center", fontsize=7.2, fontweight="bold", color=C_GT_GREEN, zorder=4)

    ax.add_patch(Rectangle((87.0, 20.5), 1.2, 1.2, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))
    ax.text(88.8, 21.1, r"$\mathrm{H\text{-}WIoU\ (IoU=0.91)}$", va="center", fontsize=7.2, fontweight="bold", color=C_PRED_BLUE, zorder=4)

    ax.add_patch(Rectangle((87.0, 13.5), 1.2, 1.2, facecolor=C_FAIL_RED, edgecolor="none", zorder=4))
    ax.text(88.8, 14.1, r"$\mathrm{Baseline\ (IoU=0.22)}$", va="center", fontsize=7.2, color=C_FAIL_RED, zorder=4)

    # ==========================================================================
    # 6. SLEEK HORIZONTAL PIPELINE FLOW ARROWS
    # ==========================================================================
    draw_styled_arrow(ax, (15.5, 50.0), (17.5, 50.0), color="#0284C7", lw=2.0, zorder=8)
    draw_styled_arrow(ax, (35.0, 50.0), (37.0, 50.0), color=C_PRED_BLUE, lw=2.0, zorder=8)
    draw_styled_arrow(ax, (60.5, 50.0), (62.5, 50.0), color=C_FLOW_PURPLE, lw=2.0, zorder=8)
    ax.text(61.5, 52.8, "RoIs", ha="center", fontsize=7.6, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)
    draw_styled_arrow(ax, (83.5, 50.0), (85.5, 50.0), color=C_AMBER_ACCENT, lw=2.0, zorder=8)

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

    print(f"[OK] Ultra-Polished Architecture Figure successfully created:\n  - PDF: {out_pdf}\n  - PNG: {out_png}\n  - SVG: {out_svg}")


if __name__ == "__main__":
    render_ultra_polished_fig5()
