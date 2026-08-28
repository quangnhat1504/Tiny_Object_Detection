"""
PaperBanana Multi-Agent Architecture Engine (Google Research Protocol arXiv:2601.23265).
Pass 2 Polish: 2-Tier Canonical Architecture Layout with Clean Inter-Module Connectors.
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

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

C_BG_CANVAS    = "#FFFFFF"
C_TEXT_DARK    = "#0F172A"
C_TEXT_MUTED   = "#334155"

P_IMAGE_BDR    = "#BAE6FD"
P_FPN_BDR      = "#BFDBFE"
P_RPN_BDR      = "#E9D5FF"
P_ROI_BDR      = "#A7F3D0"
P_HEAD_BDR     = "#FDE68A"
P_OUT_BDR      = "#FECDD3"

M_HOMOTOPY_BG  = "#F5F3FF"
M_HOMOTOPY_BDR = "#DDD6FE"

M_HLA_BG       = "#FFFBEB"
M_HLA_BDR      = "#FDE68A"

M_LOSS_BG      = "#ECFDF5"
M_LOSS_BDR     = "#A7F3D0"

C_GT_GREEN     = "#15803D"
C_PRED_BLUE    = "#1D4ED8"
C_FAIL_RED     = "#DC2626"
C_FLOW_PURPLE  = "#7C3AED"


def draw_card(ax, x, y, w, h, bg_color="#FFFFFF", border_color="#CBD5E1", radius=1.2, lw=1.2, zorder=1):
    shadow = FancyBboxPatch(
        (x + 0.25, y - 0.25), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor="#0F172A", edgecolor="none", alpha=0.035, zorder=zorder
    )
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=zorder + 1
    )
    ax.add_patch(card)
    return card


def draw_header_pill(ax, cx, top_y, title, pill_bg="#1E293B", text_color="#FFFFFF", font_size=10.0, width=None, zorder=10):
    text_len = len(title)
    w = width if width is not None else max(10.0, text_len * 0.76 + 5.0)
    h = 3.6
    x = cx - w / 2.0
    y = top_y - h / 2.0

    pill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=1.2",
        facecolor=pill_bg, edgecolor="none", zorder=zorder
    )
    ax.add_patch(pill)

    ax.text(
        cx, y + h / 2.0, title,
        ha="center", va="center",
        fontsize=font_size, fontweight="bold", color=text_color, zorder=zorder + 1
    )


def draw_styled_arrow(ax, start, end, color="#2563EB", lw=2.0, rad=0.0, dashed=False, zorder=8, head_scale=1.1):
    style = patches.ArrowStyle.Simple(head_length=3.8 * head_scale, head_width=3.0 * head_scale, tail_width=0.85 * lw)
    linestyle = "--" if dashed else "-"
    arrow = patches.FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        facecolor=color, edgecolor=color,
        linewidth=lw, linestyle=linestyle, zorder=zorder
    )
    ax.add_patch(arrow)


def draw_3d_cuboid(ax, ox, oy, w, h, depth=1.6, face_col="#3B82F6", top_col="#60A5FA", side_col="#1D4ED8", edge_col="#1E40AF", zorder=4, seed=42):
    dx = depth * 0.707
    dy = depth * 0.500

    front = Polygon(
        [[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]],
        closed=True, facecolor=face_col, edgecolor=edge_col, linewidth=0.9, zorder=zorder
    )
    ax.add_patch(front)

    rng = np.random.default_rng(seed)
    for _ in range(2):
        hx = ox + w * (0.25 + 0.5 * rng.random())
        hy = oy + h * (0.25 + 0.5 * rng.random())
        for rad, alpha, col in [(1.4, 0.35, "#F59E0B"), (0.7, 0.75, "#EF4444"), (0.3, 0.95, "#FFFFFF")]:
            ax.add_patch(Ellipse((hx, hy), rad, rad * (h / w), facecolor=col, edgecolor="none", alpha=alpha, zorder=zorder + 1))

    top = Polygon(
        [[ox, oy + h], [ox + w, oy + h], [ox + w + dx, oy + h + dy], [ox + dx, oy + h + dy]],
        closed=True, facecolor=top_col, edgecolor=edge_col, linewidth=0.9, zorder=zorder + 2
    )
    ax.add_patch(top)

    side = Polygon(
        [[ox + w, oy], [ox + w + dx, oy + dy], [ox + w + dx, oy + h + dy], [ox + w, oy + h]],
        closed=True, facecolor=side_col, edgecolor=edge_col, linewidth=0.9, zorder=zorder + 2
    )
    ax.add_patch(side)


def render_canonical_paperbanana_fig5():
    print("Executing 5-Agent PaperBanana Protocol for Figure 5 (Pass 2 Polished)...")

    fig = plt.figure(figsize=(24, 10.5), dpi=300, facecolor=C_BG_CANVAS)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    # ==========================================================================
    # TIER 1: END-TO-END DETECTOR PIPELINE (TOP TIER [y: 53.0 -> 97.0])
    # ==========================================================================
    draw_card(ax, 1.2, 53.0, 97.6, 44.0, bg_color="#F8FAFC", border_color="#CBD5E1", radius=1.6, lw=1.4, zorder=1)
    draw_header_pill(ax, 50.0, 97.5, "Tier 1: End-to-End Scale-Homotopy Object Detection Pipeline", pill_bg="#1E293B", width=48.0)

    # 1.1 Input Canvas
    draw_card(ax, 2.5, 55.5, 14.5, 38.0, bg_color="#FFFFFF", border_color=P_IMAGE_BDR, radius=1.0, lw=1.1, zorder=2)
    ax.text(9.75, 90.8, "Input Surveillance Canvas", ha="center", fontsize=8.8, fontweight="bold", color="#0369A1", zorder=3)
    ax.text(9.75, 88.0, "(1024 x 1024 Aerial View)", ha="center", fontsize=7.4, color="#0284C7", zorder=3)

    img_box = Rectangle((3.5, 66.5), 12.5, 19.5, facecolor="#E0F2FE", edgecolor="#BAE6FD", lw=1.0, zorder=3)
    ax.add_patch(img_box)
    for wy in [69.0, 73.0, 77.0, 81.0, 84.0]:
        ax.plot([4.0, 8.5], [wy, wy + 0.3], color="#BAE6FD", lw=1.0, zorder=4)
        ax.plot([10.5, 15.0], [wy - 0.2, wy + 0.2], color="#BAE6FD", lw=1.0, zorder=4)

    tx, ty = 6.2, 75.5
    ax.add_patch(Rectangle((tx - 0.5, ty - 0.7), 1.0, 1.4, facecolor="#DC2626", edgecolor="#991B1B", lw=1.2, zorder=5))
    ax.text(tx, ty - 2.0, "Micro Target", ha="center", fontsize=6.8, fontweight="bold", color="#B91C1C", zorder=6)

    loupe_cx, loupe_cy, loupe_r = 12.2, 75.5, 2.8
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty + 0.7, loupe_cy + 1.8], color="#D97706", linestyle=":", lw=1.1, zorder=6)
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty - 0.7, loupe_cy - 1.8], color="#D97706", linestyle=":", lw=1.1, zorder=6)
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r, facecolor="#FFFFFF", edgecolor="#D97706", lw=1.8, zorder=7))
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r - 0.2, facecolor="#FEF3C7", edgecolor="none", alpha=0.5, zorder=8))
    ax.add_patch(Ellipse((loupe_cx, loupe_cy + 0.9), 1.0, 1.0, facecolor="#1E3A8A", edgecolor="none", zorder=9))
    ax.add_patch(Rectangle((loupe_cx - 0.7, loupe_cy - 1.2), 1.4, 1.8, facecolor="#1D4ED8", edgecolor="none", zorder=9))
    ax.text(loupe_cx, loupe_cy - 2.0, "s = 4.8 px", ha="center", fontsize=7.0, fontweight="bold", color="#B45309", zorder=10)

    ax.text(9.75, 60.5, "Extreme Scale Imbalance", ha="center", fontsize=7.6, fontweight="bold", color="#0F172A", zorder=3)
    ax.text(9.75, 57.5, r"$\mathrm{Scale\ Range:\ }s \in [2, 32]\mathrm{px}$", ha="center", fontsize=7.2, color="#475569", zorder=3)

    # 1.2 ResNet-50 + FPN Backbone
    draw_card(ax, 19.0, 55.5, 17.5, 38.0, bg_color="#FFFFFF", border_color=P_FPN_BDR, radius=1.0, lw=1.1, zorder=2)
    ax.text(27.75, 90.8, "ResNet-50 + FPN Backbone", ha="center", fontsize=8.8, fontweight="bold", color="#1E40AF", zorder=3)
    ax.text(27.75, 88.0, "Multi-Scale Pyramid Representation", ha="center", fontsize=7.4, color="#2563EB", zorder=3)

    fpn_levels = [
        ("P5 (32x)", 20.2, 80.5, 3.2, 3.2, 1.0, "#1E3A8A", "#3B82F6", "#1D4ED8", 1),
        ("P4 (16x)", 20.2, 72.0, 4.6, 4.0, 1.2, "#1E40AF", "#60A5FA", "#2563EB", 2),
        ("P3 (8x)",  20.2, 62.5, 6.2, 5.0, 1.4, "#2563EB", "#93C5FD", "#3B82F6", 3),
        ("P2 (4x)",  20.2, 56.5, 8.0, 6.0, 1.6, "#3B82F6", "#BFDBFE", "#60A5FA", 4),
    ]
    for label, ox, oy, w, h, dep, fcol, tcol, scol, seed in fpn_levels:
        draw_3d_cuboid(ax, ox, oy, w, h, depth=dep, face_col=fcol, top_col=tcol, side_col=scol, zorder=3, seed=seed*7)
        ax.text(ox + w + dep * 0.7 + 0.8, oy + h / 2.0, label, va="center", fontsize=7.6, fontweight="bold", color="#1E293B", zorder=5)

    for i in range(len(fpn_levels) - 1):
        _, ox1, oy1, w1, h1, _, _, _, _, _ = fpn_levels[i]
        _, ox2, oy2, w2, h2, dep2, _, _, _, _ = fpn_levels[i+1]
        draw_styled_arrow(ax, (ox1 + w1/2, oy1), (ox2 + w2/2, oy2 + h2 + dep2*0.5), color="#2563EB", lw=1.4, zorder=6)

    # 1.3 Stage 1: HLA-RPN
    draw_card(ax, 38.5, 55.5, 17.5, 38.0, bg_color="#FFFFFF", border_color=P_RPN_BDR, radius=1.0, lw=1.1, zorder=2)
    ax.text(47.25, 90.8, "Stage 1: HLA-RPN", ha="center", fontsize=8.8, fontweight="bold", color="#5B21B6", zorder=3)
    ax.text(47.25, 88.0, "Scale-Homotopy Candidate Generation", ha="center", fontsize=7.4, color="#7C3AED", zorder=3)

    rpn_box = FancyBboxPatch((39.5, 68.0), 15.5, 17.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FAF5FF", edgecolor="#DDD6FE", lw=1.0, zorder=3)
    ax.add_patch(rpn_box)
    ax.text(47.25, 82.5, r"$3\times 3\ \mathrm{Conv}\ (256\text{-}\mathrm{d})$", ha="center", fontsize=7.8, fontweight="bold", color="#4C1D95", zorder=4)
    ax.text(47.25, 78.5, "Dynamic Cost Assignment:", ha="center", fontsize=7.4, color="#6D28D9", zorder=4)
    ax.text(47.25, 74.5, r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}(A_i, G_j)$", ha="center", fontsize=8.4, fontweight="bold", color="#4C1D95", zorder=4)
    ax.text(47.25, 70.5, r"$\mathrm{Top\text{-}}k\ \mathrm{Positive\ Selection}$", ha="center", fontsize=7.6, fontweight="bold", color="#15803D", zorder=4)

    ax.text(47.25, 63.5, "Zero Anchor Starvation", ha="center", fontsize=7.6, fontweight="bold", color="#166534", zorder=4)
    ax.text(47.25, 59.8, r"$\mathrm{Survival:\ } 18.2\% \to \mathbf{94.6\%}$", ha="center", fontsize=7.6, fontweight="bold", color="#15803D", zorder=4)
    ax.text(47.25, 57.0, r"$\mathrm{Generates\ 1000\ High\text{-}Quality\ RoIs}$", ha="center", fontsize=7.0, color="#475569", zorder=4)

    # 1.4 RoIAlign
    draw_card(ax, 58.0, 55.5, 12.0, 38.0, bg_color="#FFFFFF", border_color=P_ROI_BDR, radius=1.0, lw=1.1, zorder=2)
    ax.text(64.0, 90.8, "RoIAlign", ha="center", fontsize=8.8, fontweight="bold", color="#065F46", zorder=3)
    ax.text(64.0, 88.0, r"$7\times 7\times 256\ \mathrm{Pooling}$", ha="center", fontsize=7.4, color="#059669", zorder=3)

    gx_start, gy_start, g_size = 59.6, 71.5, 1.5
    for r_idx in range(4):
        for c_idx in range(4):
            ax.add_patch(Rectangle((gx_start + c_idx * (g_size + 0.3), gy_start + r_idx * (g_size + 0.3)), g_size, g_size,
                                   facecolor="#D1FAE5", edgecolor="#059669", lw=0.7, zorder=4))
    
    ax.text(64.0, 66.5, "Bilinear Sampling", ha="center", fontsize=7.6, fontweight="bold", color="#047857", zorder=4)
    ax.text(64.0, 62.5, "Continuous Sub-pixel", ha="center", fontsize=7.2, color="#065F46", zorder=4)
    ax.text(64.0, 59.0, "Fractional Grids", ha="center", fontsize=7.2, color="#065F46", zorder=4)

    # 1.5 Stage 2: Fast R-CNN Head & Bounded Loss
    draw_card(ax, 72.0, 55.5, 14.5, 38.0, bg_color="#FFFFFF", border_color=P_HEAD_BDR, radius=1.0, lw=1.1, zorder=2)
    ax.text(79.25, 90.8, "Stage 2: RoI Head", ha="center", fontsize=8.8, fontweight="bold", color="#92400E", zorder=3)
    ax.text(79.25, 88.0, "Two-Branch Prediction", ha="center", fontsize=7.4, color="#B45309", zorder=3)

    cls_box = FancyBboxPatch((73.0, 77.0), 12.5, 8.5, boxstyle="round,pad=0.0,rounding_size=0.5", facecolor="#FFFBEB", edgecolor="#FCD34D", lw=0.9, zorder=3)
    ax.add_patch(cls_box)
    ax.text(79.25, 82.5, "Classification Head", ha="center", fontsize=7.6, fontweight="bold", color="#78350F", zorder=4)
    ax.text(79.25, 79.0, r"$\mathcal{L}_{\mathrm{cls}} = \mathrm{Cross\text{-}Entropy}$", ha="center", fontsize=7.2, color="#92400E", zorder=4)

    reg_box = FancyBboxPatch((73.0, 66.0), 12.5, 8.5, boxstyle="round,pad=0.0,rounding_size=0.5", facecolor="#ECFDF5", edgecolor="#6EE7B7", lw=1.0, zorder=3)
    ax.add_patch(reg_box)
    ax.text(79.25, 71.5, "Bounded Box Head", ha="center", fontsize=7.6, fontweight="bold", color="#064E3B", zorder=4)
    ax.text(79.25, 68.0, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}$", ha="center", fontsize=7.8, fontweight="bold", color="#047857", zorder=4)

    ax.text(79.25, 61.5, "Active Gradient Optimization", ha="center", fontsize=7.4, fontweight="bold", color="#15803D", zorder=4)
    ax.text(79.25, 58.0, r"$\|\nabla_\theta \mathcal{L}\| = \mathcal{O}(1) > 0\ (\mathrm{IoU}=0)$", ha="center", fontsize=7.6, fontweight="bold", color="#DC2626", zorder=4)

    # 1.6 Output Predictions
    draw_card(ax, 88.5, 55.5, 9.3, 38.0, bg_color="#FFFFFF", border_color=P_OUT_BDR, radius=1.0, lw=1.1, zorder=2)
    ax.text(93.15, 90.8, "Final Detections", ha="center", fontsize=8.8, fontweight="bold", color="#9F1239", zorder=3)
    ax.text(93.15, 88.0, "Sub-Pixel Accurate", ha="center", fontsize=7.4, color="#BE123C", zorder=3)

    det_canvas = Rectangle((89.5, 67.5), 7.3, 17.5, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.9, zorder=3)
    ax.add_patch(det_canvas)

    ax.add_patch(Ellipse((93.15, 77.0), 1.0, 1.0, facecolor="#1E3A8A", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((92.65, 74.8), 1.0, 1.8, facecolor="#1D4ED8", edgecolor="none", zorder=4))

    ax.add_patch(Rectangle((91.5, 74.2), 3.3, 4.4, facecolor="none", edgecolor=C_GT_GREEN, lw=1.6, zorder=5))
    ax.add_patch(Rectangle((91.6, 74.3), 3.2, 4.3, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.4, zorder=6))
    ax.add_patch(Rectangle((93.8, 77.2), 3.0, 4.0, facecolor="none", edgecolor=C_FAIL_RED, lw=1.2, linestyle="--", zorder=5))

    ax.add_patch(Rectangle((89.5, 62.5), 0.9, 0.9, facecolor=C_GT_GREEN, edgecolor="none", zorder=4))
    ax.text(90.8, 63.0, "Ground Truth", va="center", fontsize=6.5, fontweight="bold", color=C_GT_GREEN, zorder=4)

    ax.add_patch(Rectangle((89.5, 59.5), 0.9, 0.9, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))
    ax.text(90.8, 60.0, "H-WIoU (0.91)", va="center", fontsize=6.5, fontweight="bold", color=C_PRED_BLUE, zorder=4)

    ax.add_patch(Rectangle((89.5, 56.5), 0.9, 0.9, facecolor=C_FAIL_RED, edgecolor="none", zorder=4))
    ax.text(90.8, 57.0, "Baseline (0.22)", va="center", fontsize=6.5, color=C_FAIL_RED, zorder=4)

    # Tier 1 Horizontal Connectors
    draw_styled_arrow(ax, (17.0, 74.5), (19.0, 74.5), color="#0284C7", lw=2.2, zorder=8)
    draw_styled_arrow(ax, (36.5, 74.5), (38.5, 74.5), color="#2563EB", lw=2.2, zorder=8)
    draw_styled_arrow(ax, (56.0, 74.5), (58.0, 74.5), color="#7C3AED", lw=2.2, zorder=8)
    ax.text(57.0, 76.5, "RoIs", ha="center", fontsize=7.2, fontweight="bold", color="#7C3AED", zorder=9)

    draw_styled_arrow(ax, (70.0, 74.5), (72.0, 74.5), color="#059669", lw=2.2, zorder=8)
    draw_styled_arrow(ax, (86.5, 74.5), (88.5, 74.5), color="#D97706", lw=2.2, zorder=8)

    # ==========================================================================
    # TIER 2: CORE MATHEMATICAL & ALGORITHMIC ENGINES (BOTTOM TIER [y: 2.5 -> 49.5])
    # ==========================================================================
    # Module A: Mathematical Homotopy & Scale Controller
    draw_card(ax, 1.2, 2.5, 32.8, 47.0, bg_color=M_HOMOTOPY_BG, border_color=M_HOMOTOPY_BDR, radius=1.4, lw=1.3, zorder=1)
    draw_header_pill(ax, 17.6, 49.5, "Module A: Mathematical Homotopy Engine", pill_bg="#4C1D95", width=30.0)

    draw_card(ax, 2.4, 30.5, 30.4, 15.0, bg_color="#FFFFFF", border_color="#DDD6FE", radius=0.9, lw=1.1, zorder=2)
    ax.text(17.6, 42.5, "Convex Scale Homotopy Metric (Theorem 1):", ha="center", fontsize=8.8, fontweight="bold", color="#4C1D95", zorder=3)
    ax.text(17.6, 37.5, r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}}(A, B) = \gamma(s_B)\,\mathrm{IoU}(A, B) + (1 - \gamma(s_B))\,\exp\left(-\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$",
            ha="center", fontsize=9.2, fontweight="bold", color="#6D28D9", zorder=3)
    ax.text(17.6, 33.0, r"$\mathrm{where}\quad \gamma(s) = \frac{s^2}{s^2 + \sigma_0^2}\in(0, 1)\quad (\sigma_0 \approx 8.0\mathrm{px})$",
            ha="center", fontsize=8.2, color="#7C3AED", zorder=3)

    sub_ax = ax.inset_axes([3.0, 4.8, 29.2, 24.2], transform=ax.transData)
    s_plot = np.linspace(0, 32, 150)
    sig0 = 8.0
    gam_plot = (s_plot**2) / (s_plot**2 + sig0**2)
    sub_ax.plot(s_plot, gam_plot, color="#7C3AED", lw=2.4, label=r"$\gamma(s) = s^2/(s^2+\sigma_0^2)$")
    sub_ax.axvline(8.0, color="#DC2626", linestyle=":", lw=1.4, label=r"$\sigma_0 = 8\mathrm{px}$")
    sub_ax.axhline(0.5, color="#D97706", linestyle=":", lw=1.1)
    sub_ax.fill_between(s_plot[s_plot <= 8.0], 0, gam_plot[s_plot <= 8.0], color="#EDE9FE", alpha=0.75, label=r"$\mathcal{W}_2\ \mathrm{Dominant}\ (s < 8\mathrm{px})$")
    sub_ax.fill_between(s_plot[s_plot >= 8.0], 0, gam_plot[s_plot >= 8.0], color="#DBEAFE", alpha=0.55, label=r"$\mathrm{IoU}\ \mathrm{Dominant}\ (s \geq 8\mathrm{px})$")
    sub_ax.set_xlim(0, 32)
    sub_ax.set_ylim(0, 1.05)
    sub_ax.set_xlabel(r"Target Scale $s$ (pixels)", fontsize=7.6, labelpad=1)
    sub_ax.set_ylabel(r"Homotopy Weight $\gamma(s)$", fontsize=7.6, labelpad=1)
    sub_ax.tick_params(labelsize=7.0, pad=1)
    sub_ax.set_facecolor("#FAFAFA")
    for spine in sub_ax.spines.values():
        spine.set_edgecolor("#DDD6FE")
        spine.set_linewidth(0.9)
    sub_ax.grid(True, linestyle="--", alpha=0.45)
    sub_ax.legend(fontsize=6.8, loc="lower right", framealpha=0.92)

    # Module B: Dynamic Homotopy Label Assigner (HLA)
    draw_card(ax, 35.2, 2.5, 31.8, 47.0, bg_color=M_HLA_BG, border_color=M_HLA_BDR, radius=1.4, lw=1.3, zorder=1)
    draw_header_pill(ax, 51.1, 49.5, "Module B: Dynamic Stage-1 Assigner (HLA)", pill_bg="#B45309", width=31.0)

    draw_card(ax, 36.4, 21.0, 14.0, 24.5, bg_color="#FEF2F2", border_color="#FECACA", radius=0.8, lw=1.0, zorder=2)
    ax.text(43.4, 42.5, "Discrete IoU Collapse", ha="center", fontsize=8.0, fontweight="bold", color="#DC2626", zorder=3)
    ax.add_patch(Rectangle((37.8, 33.5), 4.2, 4.2, facecolor="#DCFCE7", edgecolor=C_GT_GREEN, lw=1.4, zorder=4))
    ax.text(39.9, 35.6, "GT", ha="center", va="center", fontsize=7.4, fontweight="bold", color=C_GT_GREEN, zorder=5)
    ax.add_patch(Rectangle((44.2, 30.5), 4.2, 4.2, facecolor="none", edgecolor=C_FAIL_RED, lw=1.4, linestyle="--", zorder=4))
    ax.text(46.3, 32.6, "Anchor", ha="center", va="center", fontsize=7.2, fontweight="bold", color=C_FAIL_RED, zorder=5)
    ax.plot([42.2, 43.8], [35.0, 36.6], color=C_FAIL_RED, lw=2.0, zorder=6)
    ax.plot([42.2, 43.8], [36.6, 35.0], color=C_FAIL_RED, lw=2.0, zorder=6)
    ax.text(43.4, 26.5, r"$\mathrm{IoU} = 0.00 \Rightarrow \|\nabla\| \equiv 0$", ha="center", fontsize=7.6, fontweight="bold", color="#991B1B", zorder=5)
    ax.text(43.4, 23.0, "Severe Anchor Starvation", ha="center", fontsize=7.0, color="#B91C1C", zorder=5)

    draw_card(ax, 51.6, 21.0, 14.2, 24.5, bg_color="#F0FDF4", border_color="#BBF7D0", radius=0.8, lw=1.0, zorder=2)
    ax.text(58.7, 42.5, r"Continuous $\mathcal{W}_2$ Flow", ha="center", fontsize=8.0, fontweight="bold", color="#15803D", zorder=3)
    for r, a in [(3.0, 0.14), (2.0, 0.32), (1.0, 0.60)]:
        ax.add_patch(Ellipse((54.8, 35.5), r * 1.4, r * 1.4, facecolor=C_GT_GREEN, edgecolor="none", alpha=a, zorder=3))
        ax.add_patch(Ellipse((61.8, 32.5), r * 1.4, r * 1.4, facecolor=C_PRED_BLUE, edgecolor="none", alpha=a, zorder=3))
    ax.add_patch(Rectangle((53.5, 34.2), 2.6, 2.6, facecolor="none", edgecolor=C_GT_GREEN, lw=1.2, zorder=4))
    ax.add_patch(Rectangle((60.5, 31.2), 2.6, 2.6, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.2, linestyle="--", zorder=4))
    draw_styled_arrow(ax, (61.0, 32.5), (56.0, 35.0), color=C_FLOW_PURPLE, lw=2.0, zorder=6)
    ax.text(58.7, 26.5, r"$\mathcal{S}_{\mathcal{W}} > 0 \Rightarrow \|\nabla\| = \mathcal{O}(1)$", ha="center", fontsize=7.6, fontweight="bold", color="#065F46", zorder=5)
    ax.text(58.7, 23.0, "Smooth Coordinate Pull", ha="center", fontsize=7.0, color="#047857", zorder=5)

    draw_card(ax, 36.4, 4.8, 29.4, 14.5, bg_color="#FFFFFF", border_color="#FCD34D", radius=0.8, lw=1.0, zorder=2)
    ax.text(51.1, 15.5, r"Dynamic Top-$k$ HLA Positive Anchor Selection", ha="center", fontsize=8.2, fontweight="bold", color="#78350F", zorder=3)
    ax.text(51.1, 11.5, r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}(A_i, G_j) \in [0, 1]\quad \longrightarrow\quad \mathrm{Assign\ Top\text{-}}k\ \mathrm{per\ GT}$", ha="center", fontsize=8.0, fontweight="bold", color="#92400E", zorder=3)
    ax.text(51.1, 7.5, r"$\mathrm{Positive\ Anchor\ Survival:\ } 18.2\% \to \mathbf{94.6\%}\ (\mathbf{5.2\times\ Boost})$", ha="center", fontsize=7.8, fontweight="bold", color="#15803D", zorder=3)

    # Module C: Bounded Loss & Non-Vanishing Gradient
    draw_card(ax, 68.2, 2.5, 30.6, 47.0, bg_color=M_LOSS_BG, border_color=M_LOSS_BDR, radius=1.4, lw=1.3, zorder=1)
    draw_header_pill(ax, 83.5, 49.5, "Module C: Bounded Loss & Active Backprop", pill_bg="#065F46", width=30.0)

    draw_card(ax, 69.4, 30.5, 28.2, 15.0, bg_color="#FFFFFF", border_color="#A7F3D0", radius=0.8, lw=1.0, zorder=2)
    ax.text(83.5, 42.5, "Bounded Homotopy Box Loss Formulation:", ha="center", fontsize=8.6, fontweight="bold", color="#064E3B", zorder=3)
    ax.text(83.5, 37.5, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}(P_i, G_i) \in [0, 1]$", ha="center", fontsize=9.2, fontweight="bold", color="#047857", zorder=3)
    ax.text(83.5, 33.0, "Bounded Range Eliminates Gradient Explosions on Tiny Outliers", ha="center", fontsize=7.4, color="#065F46", zorder=3)

    draw_card(ax, 69.4, 4.8, 28.2, 24.2, bg_color="#FEF2F2", border_color="#FECACA", radius=0.8, lw=1.0, zorder=2)
    ax.text(83.5, 25.5, "Non-Vanishing Gradient Backpropagation:", ha="center", fontsize=8.4, fontweight="bold", color="#991B1B", zorder=3)
    
    draw_styled_arrow(ax, (93.5, 21.0), (73.5, 21.0), color=C_FAIL_RED, lw=2.6, dashed=True, zorder=5)
    ax.text(83.5, 22.5, r"$\mathrm{Active\ Gradient\ Flow:\ }\|\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}}\| = \mathcal{O}(1) > 0$", ha="center", fontsize=7.8, fontweight="bold", color="#B91C1C", zorder=6)

    ax.text(83.5, 16.5, r"$\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}} = (1-\gamma(s_B))\,\nabla_\theta \mathcal{L}_{\mathcal{W}} \neq 0\quad (\mathrm{when\ IoU} \equiv 0)$", ha="center", fontsize=8.0, fontweight="bold", color="#DC2626", zorder=4)
    ax.text(83.5, 11.8, r"$\lim_{s \to 0}(1-\gamma(s)) = 1.0 \longrightarrow \mathrm{Pure\ Optimal\ Transport\ Guidance}$", ha="center", fontsize=7.6, color="#7C3AED", zorder=4)
    ax.text(83.5, 7.5, "Guaranteed Convergence for Sub-Pixel Targets", ha="center", fontsize=7.4, fontweight="bold", color="#15803D", zorder=4)

    # Inter-Module Connections in Tier 2 (Module A feeds Module B & C)
    draw_styled_arrow(ax, (34.0, 38.0), (35.2, 38.0), color=C_FLOW_PURPLE, lw=2.2, zorder=8)
    draw_styled_arrow(ax, (67.0, 38.0), (68.2, 38.0), color="#059669", lw=2.2, zorder=8)

    # Inter-Tier Connectors (Dashed arrows from Modules to corresponding stages)
    draw_styled_arrow(ax, (51.1, 49.5), (47.25, 55.5), color="#B45309", lw=2.2, dashed=True, rad=-0.08, zorder=8)
    ax.text(49.8, 52.8, "HLA Guidance", fontsize=7.2, fontweight="bold", color="#B45309", zorder=9)

    draw_styled_arrow(ax, (83.5, 49.5), (79.25, 55.5), color="#065F46", lw=2.2, dashed=True, rad=-0.08, zorder=8)
    ax.text(82.2, 52.8, "Loss & Backprop", fontsize=7.2, fontweight="bold", color="#065F46", zorder=9)

    # Save Outputs
    out_pdf = FIG_DIR / "fig5_pipeline_architecture.pdf"
    out_png = FIG_DIR / "fig5_pipeline_architecture.png"
    out_svg = FIG_DIR / "fig5_pipeline_architecture.svg"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.savefig(out_svg, format="svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    shutil.copy(out_pdf, OUT_DIR / "fig5_pipeline_architecture.pdf")
    shutil.copy(out_png, OUT_DIR / "fig5_pipeline_architecture.png")
    shutil.copy(out_svg, OUT_DIR / "fig5_pipeline_architecture.svg")

    print(f"[OK] Canonical Figure 5 successfully created:\n  - PDF: {out_pdf}\n  - PNG: {out_png}\n  - SVG (Figma Ready): {out_svg}")


if __name__ == "__main__":
    render_canonical_paperbanana_fig5()
