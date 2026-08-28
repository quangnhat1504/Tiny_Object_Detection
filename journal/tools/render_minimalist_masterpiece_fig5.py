"""
Minimalist Academic Publication Architecture Diagram Generator (IEEE TPAMI / CVPR Standard).
Pure Geometry, Zero Text-Dumping, 3D Isometric Tensor Slices, Elegant Mathematical Hierarchy.
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
# MINIMALIST SCIENTIFIC COLOR PALETTE (Nature / CVPR Standard)
# ==============================================================================
C_BG            = "#FFFFFF"
C_TEXT_MAIN     = "#0F172A"   # Slate 900
C_TEXT_MUTED    = "#475569"   # Slate 600
C_BORDER_LIGHT  = "#E2E8F0"   # Slate 200

# Tensors & Features (Deep & Soft Blues)
C_FPN_FACE      = "#3B82F6"
C_FPN_TOP       = "#60A5FA"
C_FPN_SIDE      = "#2563EB"
C_FPN_LINE      = "#1D4ED8"

# Mechanism Accent Colors
C_GT_GREEN      = "#16A34A"   # Emerald Green
C_PRED_BLUE     = "#2563EB"   # Royal Blue
C_FAIL_RED      = "#DC2626"   # Ruby Red
C_FLOW_PURPLE   = "#7C3AED"   # Violet
C_AMBER_ACCENT  = "#D97706"   # Warm Amber


def draw_styled_arrow(ax, start, end, color="#475569", lw=1.6, rad=0.0, dashed=False, zorder=10, head_scale=1.0):
    """Draw a clean, sleek academic vector arrow."""
    style = patches.ArrowStyle.Simple(head_length=3.2 * head_scale, head_width=2.4 * head_scale, tail_width=0.7 * lw)
    linestyle = "--" if dashed else "-"
    arrow = patches.FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        facecolor=color, edgecolor=color,
        linewidth=lw, linestyle=linestyle, zorder=zorder
    )
    ax.add_patch(arrow)


def draw_isometric_feature_slice(ax, cx, cy, w, h, depth=1.2, label="", face_color="#3B82F6", alpha=0.85, zorder=4):
    """Draw an elegant semi-transparent 3D isometric feature plane with projection grid lines."""
    dx = depth * 0.707
    dy = depth * 0.500
    ox = cx - w / 2.0
    oy = cy - h / 2.0

    # Front Face
    front = Polygon(
        [[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]],
        closed=True, facecolor=face_color, edgecolor="#1E40AF", linewidth=0.8, alpha=alpha, zorder=zorder
    )
    ax.add_patch(front)

    # Top Face
    top = Polygon(
        [[ox, oy + h], [ox + w, oy + h], [ox + w + dx, oy + h + dy], [ox + dx, oy + h + dy]],
        closed=True, facecolor="#93C5FD", edgecolor="#1E40AF", linewidth=0.8, alpha=alpha * 0.9, zorder=zorder + 1
    )
    ax.add_patch(top)

    # Right Side Face
    side = Polygon(
        [[ox + w, oy], [ox + w + dx, oy + dy], [ox + w + dx, oy + h + dy], [ox + w, oy + h]],
        closed=True, facecolor="#1D4ED8", edgecolor="#1E40AF", linewidth=0.8, alpha=alpha * 0.9, zorder=zorder + 1
    )
    ax.add_patch(side)

    # Label
    if label:
        ax.text(ox + w + dx + 0.8, oy + h / 2.0 + dy / 2.0, label, va="center", fontsize=7.2, fontweight="bold", color=C_TEXT_MAIN, zorder=zorder + 2)


def render_minimalist_masterpiece():
    print("Rendering Minimalist Academic Architecture Diagram (IEEE TPAMI / CVPR Oral Standard)...")

    # High-Resolution Master Canvas (24 x 8.2 inches)
    fig = plt.figure(figsize=(24, 8.2), dpi=300, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    # ==========================================================================
    # 1. INPUT CANVAS & CIRCULAR LOUPE [x: 2.0 -> 16.0]
    # ==========================================================================
    # Clean background plate for input image
    ax.add_patch(Rectangle((2.0, 32.0), 14.0, 56.0, facecolor="#F8FAFC", edgecolor=C_BORDER_LIGHT, lw=1.0, zorder=1))
    ax.text(9.0, 84.5, "Input Aerial Image", ha="center", fontsize=9.2, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
    ax.text(9.0, 81.5, r"$1024 \times 1024\ \mathrm{px}$", ha="center", fontsize=7.6, color=C_TEXT_MUTED, zorder=3)

    # Maritime Canvas with fine wave lines
    img_rect = Rectangle((3.2, 42.0), 11.6, 36.0, facecolor="#F0F9FF", edgecolor="#BAE6FD", lw=1.0, zorder=2)
    ax.add_patch(img_rect)
    for wy in [46.0, 52.0, 58.0, 64.0, 70.0, 74.0]:
        ax.plot([4.0, 8.0], [wy, wy + 0.3], color="#E0F2FE", lw=0.8, zorder=3)
        ax.plot([9.5, 14.0], [wy - 0.2, wy + 0.2], color="#E0F2FE", lw=0.8, zorder=3)

    # Micro GT Swimmer (4x6 px target)
    tx, ty = 6.0, 58.0
    ax.add_patch(Rectangle((tx - 0.5, ty - 0.7), 1.0, 1.4, facecolor=C_FAIL_RED, edgecolor="#991B1B", lw=1.0, zorder=5))

    # Magnifying Loupe
    loupe_cx, loupe_cy, loupe_r = 11.2, 58.0, 2.6
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty + 0.7, loupe_cy + 1.8], color=C_AMBER_ACCENT, linestyle=":", lw=1.1, zorder=6)
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty - 0.7, loupe_cy - 1.8], color=C_AMBER_ACCENT, linestyle=":", lw=1.1, zorder=6)
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r, facecolor="#FFFFFF", edgecolor=C_AMBER_ACCENT, lw=1.8, zorder=7))
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r - 0.2, facecolor="#FEF3C7", edgecolor="none", alpha=0.4, zorder=8))
    ax.add_patch(Ellipse((loupe_cx, loupe_cy + 0.8), 0.9, 0.9, facecolor="#1E3A8A", edgecolor="none", zorder=9))
    ax.add_patch(Rectangle((loupe_cx - 0.6, loupe_cy - 1.1), 1.2, 1.6, facecolor=C_PRED_BLUE, edgecolor="none", zorder=9))
    ax.text(loupe_cx, loupe_cy - 1.9, r"$s = 4.8\mathrm{px}$", ha="center", fontsize=6.8, fontweight="bold", color=C_AMBER_ACCENT, zorder=10)

    ax.text(9.0, 36.0, "Tiny Object Target", ha="center", fontsize=7.6, fontweight="bold", color=C_FAIL_RED, zorder=3)

    # ==========================================================================
    # 2. RESNET-50 + FPN 3D TENSOR PYRAMID [x: 18.5 -> 36.0]
    # ==========================================================================
    ax.add_patch(Rectangle((18.5, 32.0), 17.5, 56.0, facecolor="#F8FAFC", edgecolor=C_BORDER_LIGHT, lw=1.0, zorder=1))
    ax.text(27.25, 84.5, "ResNet-50 + FPN", ha="center", fontsize=9.2, fontweight="bold", color=C_TEXT_MAIN, zorder=3)
    ax.text(27.25, 81.5, "Multi-Scale Hierarchy", ha="center", fontsize=7.6, color=C_TEXT_MUTED, zorder=3)

    fpn_layers = [
        (23.0, 73.0, 3.0, 2.6, 0.8, r"$\mathbf{P}_5\ (32\times)$", "#1E3A8A"),
        (23.0, 63.0, 4.4, 3.4, 1.0, r"$\mathbf{P}_4\ (16\times)$", "#1E40AF"),
        (23.0, 51.5, 6.0, 4.4, 1.2, r"$\mathbf{P}_3\ (8\times)$",  "#2563EB"),
        (23.0, 38.5, 7.8, 5.4, 1.4, r"$\mathbf{P}_2\ (4\times)$",  "#3B82F6"),
    ]

    for cx, cy, w, h, dep, lbl, col in fpn_layers:
        draw_isometric_feature_slice(ax, cx, cy, w, h, depth=dep, label=lbl, face_color=col, zorder=3)

    # Lateral Convolutions & Top-down Arrows
    for i in range(len(fpn_layers) - 1):
        _, cy1, _, h1, _, _, _ = fpn_layers[i]
        _, cy2, _, h2, _, _, _ = fpn_layers[i+1]
        draw_styled_arrow(ax, (23.0, cy1 - h1/2), (23.0, cy2 + h2/2 + 0.6), color=C_PRED_BLUE, lw=1.4, zorder=6)

    # Feature Output Hub
    ax.add_patch(FancyBboxPatch((28.5, 34.5), 6.5, 5.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#EFF6FF", edgecolor="#93C5FD", lw=0.9, zorder=4))
    ax.text(31.75, 38.0, r"$\mathbf{P}_2\text{--}\mathbf{P}_5$", ha="center", fontsize=7.6, fontweight="bold", color="#1D4ED8", zorder=5)
    ax.text(31.75, 35.8, r"$256\text{-}\mathrm{d}$", ha="center", fontsize=6.8, color="#2563EB", zorder=5)

    # ==========================================================================
    # 3. STAGE 1: HLA-RPN (HOMOTOPY LABEL ASSIGNMENT) [x: 38.0 -> 59.0]
    # ==========================================================================
    ax.add_patch(Rectangle((38.0, 32.0), 21.0, 56.0, facecolor="#F8FAFC", edgecolor=C_BORDER_LIGHT, lw=1.0, zorder=1))
    ax.text(48.5, 84.5, "Stage 1: HLA-RPN", ha="center", fontsize=9.2, fontweight="bold", color="#5B21B6", zorder=3)
    ax.text(48.5, 81.5, "Scale-Homotopy Candidate Generation", ha="center", fontsize=7.6, color="#7C3AED", zorder=3)

    # Comparison Graphic: (a) IoU Collapse vs (b) W2 Flow
    # (a) Discrete IoU Collapse Box
    ax.add_patch(FancyBboxPatch((39.2, 57.0), 9.0, 21.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.9, zorder=2))
    ax.text(43.7, 75.0, "Discrete IoU", ha="center", fontsize=7.6, fontweight="bold", color=C_FAIL_RED, zorder=3)
    ax.add_patch(Rectangle((40.2, 66.5), 3.0, 3.0, facecolor="#DCFCE7", edgecolor=C_GT_GREEN, lw=1.2, zorder=4))
    ax.text(41.7, 68.0, "GT", ha="center", va="center", fontsize=6.8, fontweight="bold", color=C_GT_GREEN, zorder=5)
    ax.add_patch(Rectangle((44.2, 64.0), 3.0, 3.0, facecolor="none", edgecolor=C_FAIL_RED, lw=1.2, linestyle="--", zorder=4))
    ax.text(45.7, 65.5, "Anc", ha="center", va="center", fontsize=6.5, fontweight="bold", color=C_FAIL_RED, zorder=5)
    ax.plot([42.8, 44.0], [67.0, 68.2], color=C_FAIL_RED, lw=1.8, zorder=6)
    ax.plot([42.8, 44.0], [68.2, 67.0], color=C_FAIL_RED, lw=1.8, zorder=6)
    ax.text(43.7, 60.5, r"$\mathrm{IoU} = 0 \Rightarrow \nabla \equiv 0$", ha="center", fontsize=7.2, fontweight="bold", color="#991B1B", zorder=5)
    ax.text(43.7, 58.2, "Starvation", ha="center", fontsize=6.5, color="#B91C1C", zorder=5)

    # (b) Continuous Wasserstein Flow Box
    ax.add_patch(FancyBboxPatch((49.0, 57.0), 9.0, 21.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=0.9, zorder=2))
    ax.text(53.5, 75.0, r"$\mathcal{W}_2\ \mathrm{Flow}$", ha="center", fontsize=7.6, fontweight="bold", color=C_GT_GREEN, zorder=3)
    for r, a in [(2.2, 0.15), (1.4, 0.35), (0.7, 0.7)]:
        ax.add_patch(Ellipse((51.2, 67.5), r * 1.3, r * 1.3, facecolor=C_GT_GREEN, edgecolor="none", alpha=a, zorder=3))
        ax.add_patch(Ellipse((55.8, 65.5), r * 1.3, r * 1.3, facecolor=C_PRED_BLUE, edgecolor="none", alpha=a, zorder=3))
    draw_styled_arrow(ax, (55.2, 65.5), (52.2, 67.0), color=C_FLOW_PURPLE, lw=1.8, zorder=6)
    ax.text(53.5, 60.5, r"$\mathcal{S}_{\mathcal{W}} > 0 \Rightarrow \nabla \neq 0$", ha="center", fontsize=7.2, fontweight="bold", color="#065F46", zorder=5)
    ax.text(53.5, 58.2, "Smooth Pull", ha="center", fontsize=6.5, color="#047857", zorder=5)

    # Dynamic Top-k Assignment Card
    ax.add_patch(FancyBboxPatch((39.2, 34.5), 18.8, 19.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FAF5FF", edgecolor="#DDD6FE", lw=0.9, zorder=2))
    ax.text(48.6, 50.5, "Dynamic Cost Assignment:", ha="center", fontsize=7.6, fontweight="bold", color="#4C1D95", zorder=3)
    ax.text(48.6, 45.5, r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}(A_i, G_j)$", ha="center", fontsize=8.4, fontweight="bold", color="#5B21B6", zorder=3)
    ax.text(48.6, 40.0, r"$\mathrm{Top\text{-}}k\ \mathrm{Positive\ Selection}\ (18.2\% \to \mathbf{94.6\%})$", ha="center", fontsize=7.4, fontweight="bold", color=C_GT_GREEN, zorder=3)
    ax.text(48.6, 36.5, r"$\mathrm{Generates\ 1000\ High\text{-}Quality\ RoIs}$", ha="center", fontsize=6.8, color="#6D28D9", zorder=3)

    # ==========================================================================
    # 4. ROI-ALIGN & STAGE 2 ROI HEAD [x: 61.0 -> 82.5]
    # ==========================================================================
    ax.add_patch(Rectangle((61.0, 32.0), 21.5, 56.0, facecolor="#F8FAFC", edgecolor=C_BORDER_LIGHT, lw=1.0, zorder=1))
    ax.text(71.75, 84.5, "Stage 2: RoIAlign & Head", ha="center", fontsize=9.2, fontweight="bold", color="#065F46", zorder=3)
    ax.text(71.75, 81.5, "High-Precision Boundary Regression", ha="center", fontsize=7.6, color="#059669", zorder=3)

    # RoIAlign Bilinear Grid (7x7)
    ax.add_patch(FancyBboxPatch((62.2, 57.0), 8.5, 21.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=0.9, zorder=2))
    ax.text(66.45, 75.0, "RoIAlign", ha="center", fontsize=7.8, fontweight="bold", color="#065F46", zorder=3)
    ax.text(66.45, 72.0, r"$7\times 7\times 256$", ha="center", fontsize=7.0, color="#059669", zorder=3)

    gx_start, gy_start, g_size = 63.8, 60.5, 1.0
    for r_idx in range(3):
        for c_idx in range(3):
            ax.add_patch(Rectangle((gx_start + c_idx * (g_size + 0.35), gy_start + r_idx * (g_size + 0.35)), g_size, g_size,
                                   facecolor="#D1FAE5", edgecolor="#059669", lw=0.6, zorder=4))
    ax.text(66.45, 58.2, "Bilinear Grid", ha="center", fontsize=6.5, color="#064E3B", zorder=5)

    # Two-Branch Fast R-CNN Head
    # Branch 1: Classification
    ax.add_patch(FancyBboxPatch((71.8, 68.5), 9.6, 9.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFBEB", edgecolor="#FCD34D", lw=0.9, zorder=2))
    ax.text(76.6, 74.5, "Classification Head", ha="center", fontsize=7.2, fontweight="bold", color="#78350F", zorder=3)
    ax.text(76.6, 71.0, r"$\mathcal{L}_{\mathrm{cls}} = -\sum y_c \log \hat{p}_c$", ha="center", fontsize=7.2, color="#92400E", zorder=3)

    # Branch 2: Bounded Loss Regression
    ax.add_patch(FancyBboxPatch((71.8, 57.0), 9.6, 9.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#ECFDF5", edgecolor="#059669", lw=1.1, zorder=2))
    ax.text(76.6, 63.0, "Bounded Box Loss", ha="center", fontsize=7.4, fontweight="bold", color="#064E3B", zorder=3)
    ax.text(76.6, 59.5, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}$", ha="center", fontsize=7.4, fontweight="bold", color="#047857", zorder=3)

    # Active Gradient Backprop Indicator
    ax.add_patch(FancyBboxPatch((62.2, 34.5), 19.2, 19.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.9, zorder=2))
    ax.text(71.8, 50.5, "Active Gradient Optimization:", ha="center", fontsize=7.6, fontweight="bold", color="#991B1B", zorder=3)
    draw_styled_arrow(ax, (79.0, 45.0), (64.5, 45.0), color=C_FAIL_RED, lw=2.2, dashed=True, zorder=5)
    ax.text(71.8, 46.8, r"$\|\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}}\| = \mathcal{O}(1) > 0\ (\mathrm{when\ IoU} \equiv 0)$", ha="center", fontsize=7.6, fontweight="bold", color="#DC2626", zorder=6)
    ax.text(71.8, 40.0, r"$\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}} = (1-\gamma(s_B))\,\nabla_\theta \mathcal{L}_{\mathcal{W}}$", ha="center", fontsize=7.6, fontweight="bold", color="#991B1B", zorder=4)
    ax.text(71.8, 36.5, r"$\lim_{s \to 0}(1-\gamma(s)) = 1.0 \longrightarrow \mathrm{Guaranteed\ Convergence}$", ha="center", fontsize=6.8, color="#7C3AED", zorder=4)

    # ==========================================================================
    # 5. FINAL DETECTION VERIFICATION [x: 84.5 -> 98.0]
    # ==========================================================================
    ax.add_patch(Rectangle((84.5, 32.0), 13.5, 56.0, facecolor="#F8FAFC", edgecolor=C_BORDER_LIGHT, lw=1.0, zorder=1))
    ax.text(91.25, 84.5, "Final Detections", ha="center", fontsize=9.2, fontweight="bold", color="#9F1239", zorder=3)
    ax.text(91.25, 81.5, "Sub-Pixel Accurate", ha="center", fontsize=7.6, color="#BE123C", zorder=3)

    # Detection Canvas
    det_box = Rectangle((85.8, 52.0), 10.9, 26.0, facecolor="#FFFFFF", edgecolor="#CBD5E1", lw=0.9, zorder=2)
    ax.add_patch(det_box)
    for wy in [56.0, 62.0, 68.0, 74.0]:
        ax.plot([86.5, 90.0], [wy, wy + 0.2], color="#F1F5F9", lw=0.8, zorder=3)
        ax.plot([92.0, 96.0], [wy - 0.2, wy + 0.2], color="#F1F5F9", lw=0.8, zorder=3)

    # Swimmer target
    ax.add_patch(Ellipse((91.25, 66.0), 1.0, 1.0, facecolor="#1E3A8A", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((90.75, 63.8), 1.0, 1.8, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))

    # Bounding Boxes
    ax.add_patch(Rectangle((89.6, 63.2), 3.3, 4.4, facecolor="none", edgecolor=C_GT_GREEN, lw=1.6, zorder=5)) # GT
    ax.add_patch(Rectangle((89.7, 63.3), 3.2, 4.3, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.4, zorder=6)) # H-WIoU
    ax.add_patch(Rectangle((91.8, 66.2), 3.0, 4.0, facecolor="none", edgecolor=C_FAIL_RED, lw=1.2, linestyle="--", zorder=5)) # IoU

    # Legend
    ax.add_patch(Rectangle((86.0, 44.5), 1.0, 1.0, facecolor=C_GT_GREEN, edgecolor="none", zorder=4))
    ax.text(87.6, 45.0, "Ground Truth", va="center", fontsize=6.8, fontweight="bold", color=C_GT_GREEN, zorder=4)

    ax.add_patch(Rectangle((86.0, 40.5), 1.0, 1.0, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))
    ax.text(87.6, 41.0, r"$\mathrm{H\text{-}WIoU\ (0.91)}$", va="center", fontsize=6.8, fontweight="bold", color=C_PRED_BLUE, zorder=4)

    ax.add_patch(Rectangle((86.0, 36.5), 1.0, 1.0, facecolor=C_FAIL_RED, edgecolor="none", zorder=4))
    ax.text(87.6, 37.0, r"$\mathrm{Baseline\ (0.22)}$", va="center", fontsize=6.8, color=C_FAIL_RED, zorder=4)

    # ==========================================================================
    # 6. MATHEMATICAL HOMOTOPY CONTROLLER BANNER [x: 2.0 -> 98.0, y: 3.0 -> 28.0]
    # ==========================================================================
    ax.add_patch(Rectangle((2.0, 3.0), 96.0, 26.0, facecolor="#FAF5FF", edgecolor="#DDD6FE", lw=1.2, zorder=1))
    
    # Header
    ax.text(50.0, 25.5, "Mathematical Scale-Homotopy Transition Engine (Theorem 1)", ha="center", fontsize=9.6, fontweight="bold", color="#4C1D95", zorder=3)
    
    # Formula Left
    ax.text(32.0, 18.0, r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}}(A, B) = \gamma(s_B)\,\mathrm{IoU}(A, B) + (1 - \gamma(s_B))\,\exp\left(-\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$",
            ha="center", fontsize=10.0, fontweight="bold", color="#6D28D9", zorder=3)
    ax.text(32.0, 11.5, r"$\mathrm{where}\quad \gamma(s) = \frac{s^2}{s^2 + \sigma_0^2}\in(0, 1)\quad (\sigma_0 \approx 8.0\mathrm{px}\ \mathrm{is\ the\ characteristic\ microscopic\ scale})$",
            ha="center", fontsize=8.4, color="#7C3AED", zorder=3)

    # Inset Plot Right for gamma(s) curve
    sub_ax = ax.inset_axes([64.0, 4.5, 32.0, 20.0], transform=ax.transData)
    s_plot = np.linspace(0, 32, 150)
    sig0 = 8.0
    gam_plot = (s_plot**2) / (s_plot**2 + sig0**2)
    sub_ax.plot(s_plot, gam_plot, color="#7C3AED", lw=2.2, label=r"$\gamma(s) = s^2/(s^2+\sigma_0^2)$")
    sub_ax.axvline(8.0, color=C_FAIL_RED, linestyle=":", lw=1.2, label=r"$\sigma_0 = 8\mathrm{px}$")
    sub_ax.axhline(0.5, color=C_AMBER_ACCENT, linestyle=":", lw=1.0)
    sub_ax.fill_between(s_plot[s_plot <= 8.0], 0, gam_plot[s_plot <= 8.0], color="#EDE9FE", alpha=0.75, label=r"$\mathcal{W}_2\ (s < 8\mathrm{px})$")
    sub_ax.fill_between(s_plot[s_plot >= 8.0], 0, gam_plot[s_plot >= 8.0], color="#DBEAFE", alpha=0.55, label=r"$\mathrm{IoU}\ (s \geq 8\mathrm{px})$")
    sub_ax.set_xlim(0, 32)
    sub_ax.set_ylim(0, 1.05)
    sub_ax.set_xlabel(r"Target Scale $s$ (px)", fontsize=7.2, labelpad=1)
    sub_ax.set_ylabel(r"Weight $\gamma(s)$", fontsize=7.2, labelpad=1)
    sub_ax.tick_params(labelsize=6.8, pad=1)
    sub_ax.set_facecolor("#FFFFFF")
    for spine in sub_ax.spines.values():
        spine.set_edgecolor("#DDD6FE")
        spine.set_linewidth(0.8)
    sub_ax.grid(True, linestyle="--", alpha=0.4)
    sub_ax.legend(fontsize=6.5, loc="lower right", framealpha=0.92)

    # ==========================================================================
    # 7. SLEEK PIPELINE FLOW ARROWS
    # ==========================================================================
    draw_styled_arrow(ax, (16.0, 60.0), (18.5, 60.0), color="#0284C7", lw=2.0, zorder=8)
    draw_styled_arrow(ax, (36.0, 60.0), (38.0, 60.0), color=C_PRED_BLUE, lw=2.0, zorder=8)
    draw_styled_arrow(ax, (59.0, 60.0), (61.0, 60.0), color=C_FLOW_PURPLE, lw=2.0, zorder=8)
    ax.text(60.0, 62.0, "RoIs", ha="center", fontsize=7.2, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)
    draw_styled_arrow(ax, (82.5, 60.0), (84.5, 60.0), color=C_AMBER_ACCENT, lw=2.0, zorder=8)

    # Homotopy Control Connections (Subtle dashed purple lines)
    draw_styled_arrow(ax, (50.0, 29.0), (48.5, 32.0), color=C_FLOW_PURPLE, lw=1.8, dashed=True, zorder=8)
    draw_styled_arrow(ax, (50.0, 29.0), (71.75, 32.0), color=C_FLOW_PURPLE, lw=1.8, dashed=True, rad=-0.08, zorder=8)

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

    print(f"[OK] Minimalist Masterpiece Architecture Figure successfully created:\n  - PDF: {out_pdf}\n  - PNG: {out_png}\n  - SVG: {out_svg}")


if __name__ == "__main__":
    render_minimalist_masterpiece()
