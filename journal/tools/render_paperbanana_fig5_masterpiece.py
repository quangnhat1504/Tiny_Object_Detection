"""
Masterpiece PaperBanana Figure 5 Generator (Google Research Protocol arXiv:2601.23265).
Pass 4: Masterpiece Final Perfection - Clean Horizontal Connectors & Balanced Large-Scale Typography.
"""
from __future__ import annotations
import math
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

# Universal Scientific Pastel Palette
C_BG_CANVAS    = "#FFFFFF"
C_TEXT_DARK    = "#0F172A"
C_TEXT_MUTED   = "#334155"

Z1_BG          = "#F8FAFC"
Z1_BORDER      = "#CBD5E1"
Z1_ACCENT      = "#2563EB"
Z1_PILL_BG     = "#1E3A8A"

Z2_BG          = "#FAFAFE"
Z2_BORDER      = "#E2E8F0"
Z2_AMBER_TXT   = "#B45309"
Z2_PILL_BG     = "#4C1D95"

Z3_BG          = "#F8FCF9"
Z3_BORDER      = "#D1FAE5"
Z3_ACCENT      = "#059669"
Z3_PILL_BG     = "#065F46"

Z4_BG          = "#FFF8F8"
Z4_BORDER      = "#FECDD3"
Z4_ACCENT      = "#E11D48"
Z4_PILL_BG     = "#9F1239"

C_GT_GREEN     = "#15803D"
C_PRED_BLUE    = "#1D4ED8"
C_FAIL_RED     = "#DC2626"
C_FLOW_PURPLE  = "#7C3AED"


def draw_card(ax, x, y, w, h, bg_color="#FFFFFF", border_color="#CBD5E1", radius=1.4, lw=1.2, zorder=1):
    shadow = FancyBboxPatch(
        (x + 0.3, y - 0.3), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor="#0F172A", edgecolor="none", alpha=0.04, zorder=zorder
    )
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=zorder + 1
    )
    ax.add_patch(card)
    return card


def draw_header_pill(ax, cx, top_y, title, pill_bg="#1E293B", text_color="#FFFFFF", font_size=10.5, width=None, zorder=10):
    text_len = len(title)
    w = width if width is not None else max(12.0, text_len * 0.82 + 6.0)
    h = 3.8
    x = cx - w / 2.0
    y = top_y - h / 2.0

    pill = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=1.4",
        facecolor=pill_bg, edgecolor="none", zorder=zorder
    )
    ax.add_patch(pill)

    ax.text(
        cx, y + h / 2.0, title,
        ha="center", va="center",
        fontsize=font_size, fontweight="bold", color=text_color, zorder=zorder + 1
    )


def draw_styled_arrow(ax, start, end, color="#2563EB", lw=2.2, rad=0.0, dashed=False, zorder=8, head_scale=1.2):
    style = patches.ArrowStyle.Simple(head_length=4.2 * head_scale, head_width=3.4 * head_scale, tail_width=0.9 * lw)
    linestyle = "--" if dashed else "-"
    arrow = patches.FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        facecolor=color, edgecolor=color,
        linewidth=lw, linestyle=linestyle, zorder=zorder
    )
    ax.add_patch(arrow)


def draw_3d_cuboid(ax, ox, oy, w, h, depth=2.0, face_col="#3B82F6", top_col="#60A5FA", side_col="#1D4ED8", edge_col="#1E40AF", zorder=4, seed=42):
    dx = depth * 0.707
    dy = depth * 0.500

    front = Polygon(
        [[ox, oy], [ox + w, oy], [ox + w, oy + h], [ox, oy + h]],
        closed=True, facecolor=face_col, edgecolor=edge_col, linewidth=1.0, zorder=zorder
    )
    ax.add_patch(front)

    rng = np.random.default_rng(seed)
    for _ in range(2):
        hx = ox + w * (0.25 + 0.5 * rng.random())
        hy = oy + h * (0.25 + 0.5 * rng.random())
        for rad, alpha, col in [(1.8, 0.35, "#F59E0B"), (0.9, 0.75, "#EF4444"), (0.4, 0.95, "#FFFFFF")]:
            ax.add_patch(Ellipse((hx, hy), rad, rad * (h / w), facecolor=col, edgecolor="none", alpha=alpha, zorder=zorder + 1))

    top = Polygon(
        [[ox, oy + h], [ox + w, oy + h], [ox + w + dx, oy + h + dy], [ox + dx, oy + h + dy]],
        closed=True, facecolor=top_col, edgecolor=edge_col, linewidth=1.0, zorder=zorder + 2
    )
    ax.add_patch(top)

    side = Polygon(
        [[ox + w, oy], [ox + w + dx, oy + dy], [ox + w + dx, oy + h + dy], [ox + w, oy + h]],
        closed=True, facecolor=side_col, edgecolor=edge_col, linewidth=1.0, zorder=zorder + 2
    )
    ax.add_patch(side)


def render_paperbanana_fig5():
    print("Rendering Masterpiece Figure 5 (PaperBanana Google Research Protocol - Pass 4 Master)...")

    fig = plt.figure(figsize=(24, 8.8), dpi=300, facecolor=C_BG_CANVAS)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    # 1. ZONE CONTAINERS
    draw_card(ax, 1.2, 2.5, 22.3, 94.5, bg_color=Z1_BG, border_color=Z1_BORDER, radius=1.6, lw=1.5, zorder=1)
    draw_header_pill(ax, 12.35, 97.0, "Zone 1: Input & Feature Hierarchy", pill_bg=Z1_PILL_BG, width=21.0)

    draw_card(ax, 24.8, 2.5, 30.4, 94.5, bg_color=Z2_BG, border_color=Z2_BORDER, radius=1.6, lw=1.5, zorder=1)
    draw_header_pill(ax, 40.0, 97.0, "Zone 2: Stage 1 RPN & Homotopy Assigner (HLA)", pill_bg=Z2_PILL_BG, width=29.0)

    draw_card(ax, 56.5, 2.5, 26.0, 94.5, bg_color=Z3_BG, border_color=Z3_BORDER, radius=1.6, lw=1.5, zorder=1)
    draw_header_pill(ax, 69.5, 97.0, "Zone 3: Stage 2 RoI Head & Bounded Loss", pill_bg=Z3_PILL_BG, width=25.0)

    draw_card(ax, 83.8, 2.5, 15.0, 94.5, bg_color=Z4_BG, border_color=Z4_BORDER, radius=1.6, lw=1.5, zorder=1)
    draw_header_pill(ax, 91.3, 97.0, "Zone 4: Detections", pill_bg=Z4_PILL_BG, width=14.2)

    # --------------------------------------------------------------------------
    # ZONE 1: Multi-scale Canvas, Loupe, ResNet-50 & FPN
    # --------------------------------------------------------------------------
    draw_card(ax, 2.4, 69.5, 19.9, 23.0, bg_color="#FFFFFF", border_color="#CBD5E1", radius=1.2, lw=1.1, zorder=2)
    ax.text(12.35, 90.5, "Input Surveillance Canvas (1024 x 1024)", ha="center", fontsize=9.0, fontweight="bold", color="#1E3A8A", zorder=4)

    img_box = Rectangle((3.2, 71.0), 18.3, 17.5, facecolor="#E0F2FE", edgecolor="#93C5FD", lw=1.0, zorder=3)
    ax.add_patch(img_box)
    for wy in [73.5, 76.8, 80.2, 83.8, 86.5]:
        ax.plot([3.8, 10.5], [wy, wy + 0.3], color="#BAE6FD", lw=1.2, zorder=4)
        ax.plot([12.5, 20.8], [wy - 0.2, wy + 0.2], color="#BAE6FD", lw=1.2, zorder=4)

    tx, ty = 7.5, 78.5
    ax.add_patch(Rectangle((tx - 0.6, ty - 0.8), 1.2, 1.6, facecolor="#DC2626", edgecolor="#991B1B", lw=1.4, zorder=5))
    ax.text(tx, ty - 2.2, "Micro Target", ha="center", fontsize=7.6, fontweight="bold", color="#B91C1C", zorder=6)

    loupe_cx, loupe_cy, loupe_r = 16.2, 79.5, 4.2
    ax.plot([tx + 0.6, loupe_cx - loupe_r], [ty + 0.8, loupe_cy + 2.4], color="#D97706", linestyle=":", lw=1.4, zorder=6)
    ax.plot([tx + 0.6, loupe_cx - loupe_r], [ty - 0.8, loupe_cy - 2.4], color="#D97706", linestyle=":", lw=1.4, zorder=6)

    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r, facecolor="#FFFFFF", edgecolor="#D97706", lw=2.2, zorder=7))
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r - 0.3, facecolor="#FEF3C7", edgecolor="none", alpha=0.6, zorder=8))

    ax.add_patch(Ellipse((loupe_cx, loupe_cy + 1.2), 1.4, 1.4, facecolor="#1E3A8A", edgecolor="none", zorder=9))
    ax.add_patch(Rectangle((loupe_cx - 1.0, loupe_cy - 1.8), 2.0, 2.6, facecolor="#1D4ED8", edgecolor="none", zorder=9))
    ax.text(loupe_cx, loupe_cy - 3.0, "s = 4.8 px", ha="center", fontsize=8.0, fontweight="bold", color="#B45309", zorder=10)

    draw_card(ax, 2.4, 4.5, 19.9, 62.5, bg_color="#FFFFFF", border_color="#CBD5E1", radius=1.2, lw=1.1, zorder=2)
    ax.text(12.35, 64.5, "ResNet-50 + FPN Feature Hierarchy", ha="center", fontsize=9.2, fontweight="bold", color="#1E3A8A", zorder=3)

    fpn_levels = [
        ("P5 (32x)", 4.0, 51.5, 4.5, 4.2, 1.4, "#1E3A8A", "#3B82F6", "#1D4ED8", "#93C5FD", 1),
        ("P4 (16x)", 4.0, 38.0, 6.8, 5.8, 1.8, "#1E40AF", "#60A5FA", "#2563EB", "#93C5FD", 2),
        ("P3 (8x)",  4.0, 22.5, 9.2, 7.5, 2.2, "#2563EB", "#93C5FD", "#3B82F6", "#BFDBFE", 3),
        ("P2 (4x)",  4.0, 6.5,  12.0, 9.0, 2.6, "#3B82F6", "#BFDBFE", "#60A5FA", "#DBEAFE", 4),
    ]

    for label, ox, oy, w, h, dep, fcol, tcol, scol, ecol, seed in fpn_levels:
        draw_3d_cuboid(ax, ox, oy, w, h, depth=dep, face_col=fcol, top_col=tcol, side_col=scol, edge_col=ecol, zorder=3, seed=seed*11)
        ax.text(ox + w + dep * 0.7 + 1.0, oy + h / 2.0, label, va="center", fontsize=8.2, fontweight="bold", color="#1E293B", zorder=5)

    for i in range(len(fpn_levels) - 1):
        _, ox1, oy1, w1, h1, dep1, _, _, _, _, _ = fpn_levels[i]
        _, ox2, oy2, w2, h2, dep2, _, _, _, _, _ = fpn_levels[i+1]
        draw_styled_arrow(ax, (ox1 + w1/2, oy1), (ox2 + w2/2, oy2 + h2 + dep2*0.5), color="#2563EB", lw=1.6, zorder=6)
        
        conv_badge = FancyBboxPatch((16.8, (oy1 + oy2 + h2)/2.0 - 1.3), 4.6, 2.8, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.9, zorder=6)
        ax.add_patch(conv_badge)
        ax.text(19.1, (oy1 + oy2 + h2)/2.0 + 0.1, "1x1 + 2x", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#1D4ED8", zorder=7)

    # Feature Output Hub Card in Zone 1 (Connecting smoothly to RPN)
    feat_hub = FancyBboxPatch((16.5, 4.8), 5.2, 5.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#EFF6FF", edgecolor="#3B82F6", lw=1.2, zorder=7)
    ax.add_patch(feat_hub)
    ax.text(19.1, 8.2, "P2-P5", ha="center", fontsize=7.6, fontweight="bold", color="#1E40AF", zorder=8)
    ax.text(19.1, 6.0, "Features", ha="center", fontsize=6.8, color="#2563EB", zorder=8)

    # --------------------------------------------------------------------------
    # ZONE 2: STAGE 1 RPN & HOMOTOPY ASSIGNER
    # --------------------------------------------------------------------------
    draw_card(ax, 26.0, 60.5, 13.5, 32.5, bg_color="#FEF2F2", border_color="#FECACA", radius=1.0, lw=1.1, zorder=2)
    ax.text(32.75, 90.0, "(a) Discrete IoU Collapse", ha="center", fontsize=8.6, fontweight="bold", color="#DC2626", zorder=3)

    ax.add_patch(Rectangle((27.4, 76.5), 4.4, 4.4, facecolor="#DCFCE7", edgecolor=C_GT_GREEN, lw=1.6, zorder=4))
    ax.text(29.6, 78.7, "GT", ha="center", va="center", fontsize=7.8, fontweight="bold", color=C_GT_GREEN, zorder=5)

    ax.add_patch(Rectangle((33.8, 73.0), 4.4, 4.4, facecolor="none", edgecolor=C_FAIL_RED, lw=1.6, linestyle="--", zorder=4))
    ax.text(36.0, 75.2, "Anchor", ha="center", va="center", fontsize=7.5, fontweight="bold", color=C_FAIL_RED, zorder=5)

    ax.plot([32.0, 33.6], [78.0, 79.6], color=C_FAIL_RED, lw=2.4, zorder=6)
    ax.plot([32.0, 33.6], [79.6, 78.0], color=C_FAIL_RED, lw=2.4, zorder=6)

    ax.text(32.75, 69.5, r"$\mathrm{IoU}(A, B) = 0.00$", ha="center", fontsize=8.4, fontweight="bold", color="#991B1B", zorder=5)
    ax.text(32.75, 65.5, r"$\|\nabla_\theta \mathcal{L}_{\mathrm{IoU}}\| \equiv 0\ (\mathbf{Vanished})$", ha="center", fontsize=8.0, fontweight="bold", color="#DC2626", zorder=5)
    ax.text(32.75, 62.0, "Anchor Starvation (>70% Loss)", ha="center", fontsize=7.6, fontweight="bold", color="#B91C1C", zorder=5)

    draw_card(ax, 40.5, 60.5, 13.5, 32.5, bg_color="#F0FDF4", border_color="#BBF7D0", radius=1.0, lw=1.1, zorder=2)
    ax.text(47.25, 90.0, r"(b) Continuous $\mathcal{W}_2$ Flow", ha="center", fontsize=8.6, fontweight="bold", color="#15803D", zorder=3)

    for r, a in [(3.4, 0.12), (2.4, 0.28), (1.4, 0.55)]:
        ax.add_patch(Ellipse((43.8, 78.5), r * 1.5, r * 1.5, facecolor=C_GT_GREEN, edgecolor="none", alpha=a, zorder=3))
        ax.add_patch(Ellipse((50.8, 75.0), r * 1.5, r * 1.5, facecolor=C_PRED_BLUE, edgecolor="none", alpha=a, zorder=3))

    ax.add_patch(Rectangle((42.3, 77.0), 3.0, 3.0, facecolor="none", edgecolor=C_GT_GREEN, lw=1.4, zorder=4))
    ax.add_patch(Rectangle((49.3, 73.5), 3.0, 3.0, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.4, linestyle="--", zorder=4))

    draw_styled_arrow(ax, (49.8, 75.0), (45.2, 77.8), color=C_FLOW_PURPLE, lw=2.4, zorder=6)
    ax.text(47.5, 79.2, r"$\mathcal{W}_2\ \mathrm{Flow}$", ha="center", fontsize=7.6, fontweight="bold", color=C_FLOW_PURPLE, zorder=7)

    ax.text(47.25, 69.5, r"$\mathcal{S}_{\mathcal{W}} = \exp(-\mathcal{D}_{\mathcal{W}}^2) > 0$", ha="center", fontsize=8.4, fontweight="bold", color="#065F46", zorder=5)
    ax.text(47.25, 65.5, r"$\|\nabla_\theta \mathcal{L}_{\mathcal{W}}\| = \mathcal{O}(1) > 0\ (\mathbf{Active})$", ha="center", fontsize=8.0, fontweight="bold", color="#15803D", zorder=5)
    ax.text(47.25, 62.0, "Smooth Non-Zero Coordinate Pull", ha="center", fontsize=7.6, fontweight="bold", color="#047857", zorder=5)

    draw_card(ax, 26.0, 44.5, 28.0, 14.0, bg_color="#FFFFFF", border_color="#FCD34D", radius=1.0, lw=1.3, zorder=2)
    ax.text(40.0, 55.2, "Convex Scale Homotopy Metric (Theorem 1):", ha="center", fontsize=9.0, fontweight="bold", color="#78350F", zorder=3)
    ax.text(40.0, 50.2, r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}}(A, B) = \gamma(s_B)\,\mathrm{IoU}(A, B) + (1 - \gamma(s_B))\,\exp\left(-\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$",
            ha="center", fontsize=9.6, fontweight="bold", color="#92400E", zorder=3)
    ax.text(40.0, 46.2, r"$\mathrm{where}\quad \gamma(s) = \frac{s^2}{s^2 + \sigma_0^2}\in(0, 1)\quad (\sigma_0 \approx 8.0\mathrm{px})$",
            ha="center", fontsize=8.4, color="#B45309", zorder=3)

    draw_card(ax, 26.0, 4.5, 28.0, 38.0, bg_color="#FFFFFF", border_color="#DDD6FE", radius=1.0, lw=1.2, zorder=2)
    ax.text(40.0, 39.5, "Scale Homotopy Controller & Dynamic HLA Top-k", ha="center", fontsize=9.0, fontweight="bold", color="#4C1D95", zorder=3)

    sub_ax = ax.inset_axes([27.2, 7.5, 14.8, 29.0], transform=ax.transData)
    s_plot = np.linspace(0, 32, 120)
    sig0 = 8.0
    gam_plot = (s_plot**2) / (s_plot**2 + sig0**2)
    sub_ax.plot(s_plot, gam_plot, color="#7C3AED", lw=2.4, label=r"$\gamma(s)$")
    sub_ax.axvline(8.0, color="#DC2626", linestyle=":", lw=1.4)
    sub_ax.axhline(0.5, color="#D97706", linestyle=":", lw=1.1)
    sub_ax.fill_between(s_plot[s_plot <= 8.0], 0, gam_plot[s_plot <= 8.0], color="#EDE9FE", alpha=0.75, label=r"$s < 8\mathrm{px}\ (\mathcal{W}_2)$")
    sub_ax.fill_between(s_plot[s_plot >= 8.0], 0, gam_plot[s_plot >= 8.0], color="#DBEAFE", alpha=0.55, label=r"$s \geq 8\mathrm{px}\ (\mathrm{IoU})$")
    sub_ax.set_xlim(0, 32)
    sub_ax.set_ylim(0, 1.05)
    sub_ax.set_xlabel(r"Target Scale $s$ (px)", fontsize=7.6, labelpad=1)
    sub_ax.set_ylabel(r"Homotopy Weight $\gamma(s)$", fontsize=7.6, labelpad=1)
    sub_ax.tick_params(labelsize=7.0, pad=1)
    sub_ax.set_facecolor("#FAFAFA")
    for spine in sub_ax.spines.values():
        spine.set_edgecolor("#DDD6FE")
        spine.set_linewidth(1.0)
    sub_ax.grid(True, linestyle="--", alpha=0.45)
    sub_ax.legend(fontsize=6.5, loc="lower right", framealpha=0.92)

    hla_box = FancyBboxPatch((43.0, 7.5), 10.0, 29.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#F5F3FF", edgecolor="#DDD6FE", lw=1.1, zorder=3)
    ax.add_patch(hla_box)
    ax.text(48.0, 33.5, "Dynamic Top-k HLA", ha="center", fontsize=8.2, fontweight="bold", color="#5B21B6", zorder=4)
    ax.text(48.0, 28.5, "Cost Matrix:", ha="center", fontsize=7.6, color="#4C1D95", zorder=4)
    ax.text(48.0, 24.2, r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}$", ha="center", fontsize=8.4, fontweight="bold", color="#6D28D9", zorder=4)
    ax.text(48.0, 18.0, "Anchor Survival:", ha="center", fontsize=7.6, color="#4C1D95", zorder=4)
    ax.text(48.0, 13.8, r"$0.18 \to \mathbf{0.94}\ (\mathbf{5.2\times})$", ha="center", fontsize=8.6, fontweight="bold", color="#15803D", zorder=4)
    ax.text(48.0, 9.5, "Starvation Eliminated", ha="center", fontsize=7.2, fontweight="bold", color="#166534", zorder=4)

    # --------------------------------------------------------------------------
    # ZONE 3: STAGE 2 ROI HEAD, ROI-ALIGN, BOUNDED LOSS & BACKPROP
    # --------------------------------------------------------------------------
    draw_card(ax, 57.8, 69.5, 23.4, 23.0, bg_color="#FFFFFF", border_color="#A7F3D0", radius=1.0, lw=1.2, zorder=2)
    ax.text(69.5, 89.5, r"$\mathrm{RoIAlign\ Feature\ Pooling}\ (7\times 7\times 256)$", ha="center", fontsize=8.8, fontweight="bold", color="#065F46", zorder=3)

    gx_start, gy_start, g_size = 60.0, 72.5, 1.5
    for r_idx in range(4):
        for c_idx in range(4):
            ax.add_patch(Rectangle((gx_start + c_idx * (g_size + 0.35), gy_start + r_idx * (g_size + 0.35)), g_size, g_size,
                                   facecolor="#D1FAE5", edgecolor="#059669", lw=0.8, zorder=4))
    
    ax.text(69.8, 80.5, "Bilinear Sampling Grid", fontsize=8.2, fontweight="bold", color="#047857", zorder=5)
    ax.text(69.8, 76.5, "Continuous Sub-pixel Interpolation", fontsize=7.6, color="#065F46", zorder=5)
    ax.text(69.8, 73.0, "High-Resolution Boundary Encoding", fontsize=7.4, color="#475569", zorder=5)

    draw_card(ax, 57.8, 41.5, 23.4, 25.5, bg_color="#FFFFFF", border_color="#A7F3D0", radius=1.0, lw=1.2, zorder=2)
    
    ax.add_patch(FancyBboxPatch((59.0, 55.0), 21.0, 10.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#F0FDF4", edgecolor="#86EFAC", lw=1.0, zorder=3))
    ax.text(69.5, 62.2, r"$\mathrm{Classification\ Head}\ (2\times \mathrm{FC}\ 1024)$", ha="center", fontsize=8.0, fontweight="bold", color="#065F46", zorder=4)
    ax.text(69.5, 57.8, r"$\mathcal{L}_{\mathrm{cls}} = -\sum y_c \log \hat{p}_c\quad (\mathrm{Cross\text{-}Entropy})$", ha="center", fontsize=7.8, color="#15803D", zorder=4)

    ax.add_patch(FancyBboxPatch((59.0, 43.5), 21.0, 10.0, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#ECFDF5", edgecolor="#059669", lw=1.2, zorder=3))
    ax.text(69.5, 50.5, "Bounded Homotopy Box Loss Branch", ha="center", fontsize=8.2, fontweight="bold", color="#064E3B", zorder=4)
    ax.text(69.5, 46.2, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}(P_i, G_i) \in [0, 1]$", ha="center", fontsize=8.2, fontweight="bold", color="#047857", zorder=4)

    draw_card(ax, 57.8, 4.5, 23.4, 34.5, bg_color="#FEF2F2", border_color="#FECACA", radius=1.0, lw=1.2, zorder=2)
    ax.text(69.5, 35.5, "Non-Vanishing Backprop Optimization", ha="center", fontsize=8.8, fontweight="bold", color="#991B1B", zorder=3)

    draw_styled_arrow(ax, (78.0, 29.5), (61.0, 29.5), color=C_FAIL_RED, lw=2.8, dashed=True, zorder=5)
    ax.text(69.5, 31.8, r"$\mathrm{Active\ Gradient\ Flow:\ }\|\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}}\| = \mathcal{O}(1) > 0$", ha="center", fontsize=8.2, fontweight="bold", color="#B91C1C", zorder=6)

    grad_box = FancyBboxPatch((59.0, 7.5), 21.0, 19.5, boxstyle="round,pad=0.0,rounding_size=0.8", facecolor="#FFFFFF", edgecolor="#FCA5A5", lw=1.0, zorder=3)
    ax.add_patch(grad_box)
    ax.text(69.5, 23.5, "Gradient Preservation Under IoU = 0:", ha="center", fontsize=7.8, fontweight="bold", color="#991B1B", zorder=4)
    ax.text(69.5, 18.8, r"$\nabla_\theta \mathcal{L}_{\mathrm{H\text{-}WIoU}} = (1-\gamma(s_B))\,\nabla_\theta \mathcal{L}_{\mathcal{W}}$", ha="center", fontsize=8.2, fontweight="bold", color="#DC2626", zorder=4)
    ax.text(69.5, 13.8, r"$\lim_{s \to 0}(1-\gamma(s)) = 1.0 \rightarrow \mathrm{Pure\ Optimal\ Transport}$", ha="center", fontsize=7.6, color="#7C3AED", zorder=4)
    ax.text(69.5, 9.5, "Guaranteed Convergence on Microscopic Targets", ha="center", fontsize=7.4, fontweight="bold", color="#15803D", zorder=4)

    # --------------------------------------------------------------------------
    # ZONE 4: LOCALIZATION QUALITY & BENCHMARK VERIFICATION
    # --------------------------------------------------------------------------
    draw_card(ax, 84.8, 49.5, 13.0, 43.0, bg_color="#FFFFFF", border_color="#FECDD3", radius=1.0, lw=1.2, zorder=2)
    ax.text(91.3, 89.5, "Localization Quality", ha="center", fontsize=8.6, fontweight="bold", color="#9F1239", zorder=3)

    det_rect = Rectangle((86.0, 65.5), 10.6, 21.0, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.0, zorder=3)
    ax.add_patch(det_rect)

    for wy in [68.0, 72.0, 76.0, 80.0, 84.0]:
        ax.plot([86.5, 90.0], [wy, wy + 0.3], color="#E2E8F0", lw=0.8, zorder=3)
        ax.plot([92.0, 96.0], [wy - 0.2, wy + 0.2], color="#E2E8F0", lw=0.8, zorder=3)

    ax.add_patch(Ellipse((90.5, 76.5), 1.2, 1.2, facecolor="#1E3A8A", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((89.8, 73.8), 1.4, 2.2, facecolor="#1D4ED8", edgecolor="none", zorder=4))

    ax.add_patch(Rectangle((88.6, 73.2), 3.8, 5.0, facecolor="none", edgecolor=C_GT_GREEN, lw=1.8, zorder=5))
    ax.add_patch(Rectangle((88.7, 73.3), 3.7, 4.9, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.6, zorder=6))
    ax.add_patch(Rectangle((91.2, 76.8), 3.6, 4.6, facecolor="none", edgecolor=C_FAIL_RED, lw=1.4, linestyle="--", zorder=5))

    ax.add_patch(Rectangle((86.4, 60.5), 1.2, 1.2, facecolor=C_GT_GREEN, edgecolor="none", zorder=4))
    ax.text(88.2, 61.1, "Ground Truth", va="center", fontsize=7.2, fontweight="bold", color=C_GT_GREEN, zorder=4)

    ax.add_patch(Rectangle((86.4, 56.5), 1.2, 1.2, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))
    ax.text(88.2, 57.1, r"$\mathrm{H\text{-}WIoU\ (IoU=0.91)}$", va="center", fontsize=7.2, fontweight="bold", color=C_PRED_BLUE, zorder=4)

    ax.add_patch(Rectangle((86.4, 52.5), 1.2, 1.2, facecolor=C_FAIL_RED, edgecolor="none", zorder=4))
    ax.text(88.2, 53.1, r"$\mathrm{Baseline\ (IoU=0.22)}$", va="center", fontsize=7.2, color=C_FAIL_RED, zorder=4)

    draw_card(ax, 84.8, 4.5, 13.0, 42.5, bg_color="#FFFFFF", border_color="#FECDD3", radius=1.0, lw=1.2, zorder=2)
    ax.text(91.3, 43.5, "Benchmark Verification", ha="center", fontsize=8.4, fontweight="bold", color="#9F1239", zorder=3)

    tp_pill = FancyBboxPatch((85.6, 30.5), 11.4, 10.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.9, zorder=3)
    ax.add_patch(tp_pill)
    ax.text(91.3, 38.0, "TinyPerson Test", ha="center", fontsize=7.6, fontweight="bold", color="#991B1B", zorder=4)
    ax.text(91.3, 34.2, r"$\mathrm{AP}^{0.50}_{\mathrm{all}}: \mathbf{23.77\%}\ (+2.54\%)$", ha="center", fontsize=7.2, fontweight="bold", color="#B91C1C", zorder=4)
    ax.text(91.3, 31.5, r"$\mathrm{AP}^{0.25}_{\mathrm{all}}: \mathbf{48.58\%}\ (+3.17\%)$", ha="center", fontsize=7.0, color="#991B1B", zorder=4)

    aitod_pill = FancyBboxPatch((85.6, 17.5), 11.4, 11.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.9, zorder=3)
    ax.add_patch(aitod_pill)
    ax.text(91.3, 25.5, "AI-TOD-v2 Test", ha="center", fontsize=7.6, fontweight="bold", color="#1E40AF", zorder=4)
    ax.text(91.3, 21.8, r"$\mathrm{AR}_{1500}: \mathbf{26.34\%}$", ha="center", fontsize=7.2, fontweight="bold", color="#1D4ED8", zorder=4)
    ax.text(91.3, 19.2, r"$\mathrm{AP}_{vt}: \mathbf{5.20\%}\ (+24.1\%)$", ha="center", fontsize=7.0, color="#1E3A8A", zorder=4)

    eff_pill = FancyBboxPatch((85.6, 6.8), 11.4, 9.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=0.9, zorder=3)
    ax.add_patch(eff_pill)
    ax.text(91.3, 12.8, "Zero Parameter Bloat", ha="center", fontsize=7.6, fontweight="bold", color="#065F46", zorder=4)
    ax.text(91.3, 9.0, r"$+0\ \mathrm{Params}\ |\ 54.4\ \mathrm{FPS}$", ha="center", fontsize=7.2, fontweight="bold", color="#15803D", zorder=4)

    # --------------------------------------------------------------------------
    # PIPELINE DATA FLOW ARROWS (CRITIC PASS 4 - PERFECT ALIGNMENT)
    # --------------------------------------------------------------------------
    # 1. Image -> Backbone
    draw_styled_arrow(ax, (12.35, 69.5), (12.35, 67.0), color=Z1_ACCENT, lw=2.0, zorder=8)

    # 2. FPN Hub -> Stage 1 RPN (Clean horizontal connector at bottom)
    draw_styled_arrow(ax, (21.7, 7.3), (26.0, 7.3), color=Z1_ACCENT, lw=2.4, zorder=8)

    # 3. RPN HLA Proposals -> RoIAlign (Top right of Zone 2 to Top left of Zone 3)
    draw_styled_arrow(ax, (54.0, 77.0), (57.8, 77.0), color=Z2_AMBER_TXT, lw=2.4, zorder=8)
    ax.text(55.9, 79.5, "HLA RoIs", ha="center", fontsize=7.8, fontweight="bold", color=Z2_AMBER_TXT, zorder=9)

    # 4. RoIAlign -> Heads
    draw_styled_arrow(ax, (69.5, 69.5), (69.5, 67.0), color=Z3_ACCENT, lw=2.0, zorder=8)

    # 5. RoI Head -> Detections (Clean horizontal flow into Zone 4)
    draw_styled_arrow(ax, (81.2, 71.0), (84.8, 71.0), color=Z4_ACCENT, lw=2.4, zorder=8)
    ax.text(83.0, 73.5, "Boxes", ha="center", fontsize=7.8, fontweight="bold", color=Z4_ACCENT, zorder=9)

    # 6. Scale Homotopy Control Flow (Vertical dashed purple arrow)
    draw_styled_arrow(ax, (40.0, 44.5), (40.0, 42.5), color=C_FLOW_PURPLE, lw=2.0, dashed=True, zorder=8)

    # 7. Homotopy loss modulation to RoI Head (Dashed purple horizontal arrow)
    draw_styled_arrow(ax, (54.0, 48.5), (59.0, 48.5), color=C_FLOW_PURPLE, lw=2.0, dashed=True, zorder=8)
    ax.text(56.5, 50.8, r"$\gamma(s)$", ha="center", fontsize=8.0, fontweight="bold", color=C_FLOW_PURPLE,
            bbox=dict(boxstyle="circle,pad=0.2", facecolor="#EDE9FE", edgecolor="#8B5CF6", lw=0.9), zorder=9)

    out_pdf = FIG_DIR / "fig5_pipeline_architecture.pdf"
    out_png = FIG_DIR / "fig5_pipeline_architecture.png"
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.savefig(out_png, format="png", bbox_inches="tight", pad_inches=0.03, dpi=300)
    plt.close(fig)

    shutil.copy(out_pdf, OUT_DIR / "fig5_pipeline_architecture.pdf")
    shutil.copy(out_png, OUT_DIR / "fig5_pipeline_architecture.png")

    print(f"[OK] Masterpiece Figure 5 (Pass 4 Master) successfully created:\n  - PDF: {out_pdf}\n  - PNG: {out_png}")


if __name__ == "__main__":
    render_paperbanana_fig5()
