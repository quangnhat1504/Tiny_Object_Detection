"""
FAIR / DeepMind Pure Academic Architecture Generator (CVPR / IEEE TPAMI Gold Standard).
100% Pure Architecture: Zero Benchmark Score Contamination, Exact Tensor Flow, Clean Geometry.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, Ellipse, Rectangle, Circle
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
# RESTRAINED SCIENTIFIC PALETTE (DeepMind / FAIR Standard)
# ==============================================================================
C_BG            = "#FFFFFF"
C_TEXT_MAIN     = "#0F172A"   # Slate 900
C_TEXT_MUTED    = "#64748B"   # Slate 500
C_BORDER_LIGHT  = "#CBD5E1"   # Slate 300

# Mechanism Accent Colors
C_GT_GREEN      = "#15803D"   # Emerald Green
C_PRED_BLUE     = "#2563EB"   # Royal Blue
C_FAIL_RED      = "#DC2626"   # Ruby Red
C_FLOW_PURPLE   = "#7C3AED"   # Violet
C_AMBER_ACCENT  = "#D97706"   # Warm Amber


def draw_styled_arrow(ax, start, end, color="#475569", lw=1.5, dashed=False, zorder=10):
    """Draw a clean, sleek academic vector arrow."""
    linestyle = "--" if dashed else "-"
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=linestyle,
            mutation_scale=11,
            shrinkA=0, shrinkB=0,
        ),
        zorder=zorder
    )


def draw_module_box(ax, x, y, w, h, title="", subtitle="", bg_color="#F8FAFC", border_color="#CBD5E1", lw=1.1, radius=0.8, zorder=2):
    """Draw a clean scientific module card."""
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.0,rounding_size={radius}", facecolor=bg_color, edgecolor=border_color, lw=lw, zorder=zorder)
    ax.add_patch(box)
    if title:
        ax.text(x + w / 2.0, y + h - 3.2, title, ha="center", va="center", fontsize=8.4, fontweight="bold", color=C_TEXT_MAIN, zorder=zorder + 1)
    if subtitle:
        ax.text(x + w / 2.0, y + h - 6.2, subtitle, ha="center", va="center", fontsize=7.0, color=C_TEXT_MUTED, zorder=zorder + 1)
    return box


def render_pure_minimalist_fig5():
    print("[FAIR/DeepMind Standard] Rendering Pure Minimalist Architecture Figure (Zero Score Contamination)...")

    # Canonical Compact Figure Dimensions (15.0 x 5.2 inches)
    fig = plt.figure(figsize=(15.0, 5.2), dpi=300, facecolor=C_BG)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    # ==========================================================================
    # 1. INPUT IMAGE & TINY OBJECT ZOOM [x: 2.0 -> 14.0, y: 32.0 -> 94.0]
    # ==========================================================================
    draw_module_box(ax, 2.0, 32.0, 12.0, 62.0, title="Input Image", subtitle=r"$1024 \times 1024$", bg_color="#FFFFFF", border_color=C_BORDER_LIGHT)

    # Image Canvas
    ax.add_patch(Rectangle((3.2, 43.0), 9.6, 40.0, facecolor="#F0F9FF", edgecolor="#BAE6FD", lw=0.8, zorder=3))
    for wy in [48.0, 54.0, 60.0, 66.0, 72.0, 78.0]:
        ax.plot([4.0, 7.0], [wy, wy + 0.3], color="#E0F2FE", lw=0.7, zorder=4)
        ax.plot([8.5, 11.5], [wy - 0.2, wy + 0.2], color="#E0F2FE", lw=0.7, zorder=4)

    # Micro Target & Loupe
    tx, ty = 5.2, 62.0
    ax.add_patch(Rectangle((tx - 0.5, ty - 0.7), 1.0, 1.4, facecolor=C_FAIL_RED, edgecolor="#991B1B", lw=1.0, zorder=5))

    loupe_cx, loupe_cy, loupe_r = 9.2, 62.0, 2.4
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty + 0.7, loupe_cy + 1.6], color=C_AMBER_ACCENT, linestyle=":", lw=1.0, zorder=6)
    ax.plot([tx + 0.5, loupe_cx - loupe_r], [ty - 0.7, loupe_cy - 1.6], color=C_AMBER_ACCENT, linestyle=":", lw=1.0, zorder=6)
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r, facecolor="#FFFFFF", edgecolor=C_AMBER_ACCENT, lw=1.5, zorder=7))
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r - 0.2, facecolor="#FEF3C7", edgecolor="none", alpha=0.4, zorder=8))
    ax.add_patch(Ellipse((loupe_cx, loupe_cy + 0.6), 0.7, 0.7, facecolor="#1E3A8A", edgecolor="none", zorder=9))
    ax.add_patch(Rectangle((loupe_cx - 0.5, loupe_cy - 0.9), 1.0, 1.3, facecolor=C_PRED_BLUE, edgecolor="none", zorder=9))
    ax.text(loupe_cx, loupe_cy - 1.6, r"$s = 4.8\mathrm{px}$", ha="center", fontsize=6.2, fontweight="bold", color=C_AMBER_ACCENT, zorder=10)

    ax.text(8.0, 37.0, "Micro Target", ha="center", fontsize=7.2, fontweight="bold", color=C_FAIL_RED, zorder=3)
    ax.text(8.0, 34.2, r"$(s < 8\mathrm{px})$", ha="center", fontsize=6.5, color=C_TEXT_MUTED, zorder=3)

    # ==========================================================================
    # 2. RESNET-50 + FPN [x: 16.5 -> 31.0, y: 32.0 -> 94.0]
    # ==========================================================================
    draw_module_box(ax, 16.5, 32.0, 14.5, 62.0, title="ResNet-50 + FPN", subtitle="Feature Hierarchy", bg_color="#F8FAFC", border_color=C_BORDER_LIGHT)

    fpn_levels = [
        (22.5, 78.0, 3.2, 2.4, r"$\mathbf{P}_5\ (32\times)$", "#1E3A8A"),
        (22.5, 67.0, 4.4, 3.2, r"$\mathbf{P}_4\ (16\times)$", "#1E40AF"),
        (22.5, 54.0, 5.8, 4.0, r"$\mathbf{P}_3\ (8\times)$",  "#2563EB"),
        (22.5, 39.5, 7.4, 5.2, r"$\mathbf{P}_2\ (4\times)$",  "#3B82F6"),
    ]

    for cx, cy, w, h, lbl, col in fpn_levels:
        ax.add_patch(Rectangle((cx - w/2, cy - h/2), w, h, facecolor=col, edgecolor="#1E40AF", lw=0.8, zorder=4))
        ax.text(cx + w/2 + 1.2, cy, lbl, va="center", fontsize=7.0, fontweight="bold", color=C_TEXT_MAIN, zorder=5)

    for i in range(len(fpn_levels) - 1):
        _, cy1, _, h1, _, _ = fpn_levels[i]
        _, cy2, _, h2, _, _ = fpn_levels[i+1]
        draw_styled_arrow(ax, (22.5, cy1 - h1/2), (22.5, cy2 + h2/2 + 0.5), color=C_PRED_BLUE, lw=1.2, zorder=6)

    # ==========================================================================
    # 3. STAGE 1: HLA-RPN [x: 33.5 -> 51.5, y: 32.0 -> 94.0]
    # ==========================================================================
    draw_module_box(ax, 33.5, 32.0, 18.0, 62.0, title="Stage 1: HLA-RPN", subtitle="Candidate Generation", bg_color="#F8FAFC", border_color=C_BORDER_LIGHT)

    # Dynamic Cost Matching Box (Pure Architecture)
    ax.add_patch(FancyBboxPatch((34.8, 61.0), 15.4, 25.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FAF5FF", edgecolor="#DDD6FE", lw=0.9, zorder=3))
    ax.text(42.5, 80.5, "Dynamic Cost Matching", ha="center", fontsize=7.4, fontweight="bold", color="#5B21B6", zorder=4)
    ax.text(42.5, 73.5, r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}(A_i, G_j)$", ha="center", fontsize=8.0, fontweight="bold", color="#6D28D9", zorder=4)
    ax.text(42.5, 66.5, r"$\mathrm{Top\text{-}}k\ \mathrm{Positive\ Selection}$", ha="center", fontsize=7.0, fontweight="bold", color=C_GT_GREEN, zorder=4)

    # Proposals Box (Pure Architecture)
    ax.add_patch(FancyBboxPatch((34.8, 38.0), 15.4, 19.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=0.9, zorder=3))
    ax.text(42.5, 50.5, "Candidate Proposals (RoIs)", ha="center", fontsize=7.4, fontweight="bold", color="#166534", zorder=4)
    ax.text(42.5, 44.5, r"$\mathcal{L}_{\mathrm{rpn}} = \mathcal{L}_{\mathrm{cls}} + \lambda \mathcal{L}_{\mathrm{reg}}$", ha="center", fontsize=7.0, color="#15803D", zorder=4)

    # ==========================================================================
    # 4. STAGE 2: ROI-ALIGN & ROI HEAD [x: 54.0 -> 78.5, y: 32.0 -> 94.0]
    # ==========================================================================
    draw_module_box(ax, 54.0, 32.0, 24.5, 62.0, title="Stage 2: RoIAlign & Head", subtitle="Classification & Bounded Loss", bg_color="#F8FAFC", border_color=C_BORDER_LIGHT)

    # RoIAlign Grid
    ax.add_patch(FancyBboxPatch((55.2, 58.0), 8.5, 28.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#ECFDF5", edgecolor="#A7F3D0", lw=0.9, zorder=3))
    ax.text(59.45, 80.5, "RoIAlign", ha="center", fontsize=7.6, fontweight="bold", color="#065F46", zorder=4)
    ax.text(59.45, 76.0, r"$7\times 7\times 256$", ha="center", fontsize=6.8, color="#059669", zorder=4)
    gx_start, gy_start, g_size = 56.6, 62.0, 1.1
    for r_idx in range(3):
        for c_idx in range(3):
            ax.add_patch(Rectangle((gx_start + c_idx * (g_size + 0.35), gy_start + r_idx * (g_size + 0.35)), g_size, g_size,
                                   facecolor="#D1FAE5", edgecolor="#059669", lw=0.6, zorder=5))
    ax.text(59.45, 59.5, "Bilinear Sampling", ha="center", fontsize=6.0, color="#064E3B", zorder=5)

    # Two-Branch Head
    ax.add_patch(FancyBboxPatch((65.0, 72.5), 12.2, 13.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFBEB", edgecolor="#FCD34D", lw=0.9, zorder=3))
    ax.text(71.1, 81.0, "Classification Head", ha="center", fontsize=7.2, fontweight="bold", color="#78350F", zorder=4)
    ax.text(71.1, 76.0, r"$\mathcal{L}_{\mathrm{cls}} = \mathrm{Cross\text{-}Entropy}$", ha="center", fontsize=6.8, color="#92400E", zorder=4)

    ax.add_patch(FancyBboxPatch((65.0, 56.5), 12.2, 13.5, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FFFFFF", edgecolor="#059669", lw=1.2, zorder=3))
    ax.text(71.1, 65.0, "Bounded Box Head", ha="center", fontsize=7.2, fontweight="bold", color="#064E3B", zorder=4)
    ax.text(71.1, 60.0, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}$", ha="center", fontsize=7.2, fontweight="bold", color="#047857", zorder=4)

    # Active Backprop Box
    ax.add_patch(FancyBboxPatch((55.2, 38.0), 22.0, 15.0, boxstyle="round,pad=0.0,rounding_size=0.6", facecolor="#FEF2F2", edgecolor="#FECACA", lw=0.9, zorder=3))
    ax.text(66.2, 48.0, r"$\mathrm{Active\ Gradient:\ }\|\nabla_\theta \mathcal{L}\| = \mathcal{O}(1) > 0\ (\mathrm{IoU}\equiv 0)$", ha="center", fontsize=7.0, fontweight="bold", color="#DC2626", zorder=4)
    ax.text(66.2, 42.0, r"$\lim_{s \to 0}(1-\gamma(s)) = 1.0 \longrightarrow \mathrm{Guaranteed\ Convergence}$", ha="center", fontsize=6.6, color="#7C3AED", zorder=4)

    # ==========================================================================
    # 5. FINAL DETECTIONS [x: 81.0 -> 98.0, y: 32.0 -> 94.0]
    # ==========================================================================
    draw_module_box(ax, 81.0, 32.0, 17.0, 62.0, title="Final Detections", subtitle="Sub-Pixel Precision", bg_color="#FFFFFF", border_color=C_BORDER_LIGHT)

    det_canvas = Rectangle((82.5, 52.0), 14.0, 30.0, facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8, zorder=3)
    ax.add_patch(det_canvas)

    ax.add_patch(Ellipse((88.5, 67.0), 1.0, 1.0, facecolor="#1E3A8A", edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((88.0, 64.8), 1.0, 1.8, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))

    # Bounding Boxes
    ax.add_patch(Rectangle((86.8, 64.2), 3.4, 4.4, facecolor="none", edgecolor=C_GT_GREEN, lw=1.6, zorder=5)) # GT
    ax.add_patch(Rectangle((86.9, 64.3), 3.3, 4.3, facecolor="none", edgecolor=C_PRED_BLUE, lw=1.4, zorder=6)) # H-WIoU
    ax.add_patch(Rectangle((89.2, 67.2), 3.0, 4.0, facecolor="none", edgecolor=C_FAIL_RED, lw=1.2, linestyle="--", zorder=5)) # Baseline

    # Clean Legend (Zero Benchmark Contamination)
    ax.add_patch(Rectangle((83.5, 45.0), 1.0, 1.0, facecolor=C_GT_GREEN, edgecolor="none", zorder=4))
    ax.text(85.2, 45.5, "Ground Truth", va="center", fontsize=6.8, fontweight="bold", color=C_GT_GREEN, zorder=4)

    ax.add_patch(Rectangle((83.5, 40.5), 1.0, 1.0, facecolor=C_PRED_BLUE, edgecolor="none", zorder=4))
    ax.text(85.2, 41.0, r"$\mathrm{H\text{-}WIoU\ (Predicted)}$", va="center", fontsize=6.8, fontweight="bold", color=C_PRED_BLUE, zorder=4)

    ax.add_patch(Rectangle((83.5, 36.0), 1.0, 1.0, facecolor=C_FAIL_RED, edgecolor="none", zorder=4))
    ax.text(85.2, 36.5, r"$\mathrm{Baseline\ (Predicted)}$", va="center", fontsize=6.8, color=C_FAIL_RED, zorder=4)

    # ==========================================================================
    # 6. THE PROPOSED CORE HUB [x: 33.5 -> 78.5, y: 4.0 -> 26.0]
    # ==========================================================================
    draw_module_box(ax, 33.5, 4.0, 45.0, 22.0, title="Scale-Aware Homotopy Engine (Proposed Core Theorem 1)", bg_color="#FAF5FF", border_color="#8B5CF6", lw=1.3, radius=1.0)

    ax.text(56.0, 15.5, r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}}(A, B) = \gamma(s_B)\,\mathrm{IoU}(A, B) + (1 - \gamma(s_B))\,\exp\left(-\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$",
            ha="center", fontsize=8.6, fontweight="bold", color="#6D28D9", zorder=4)
    ax.text(56.0, 9.0, r"$\mathrm{where}\quad \gamma(s) = \frac{s^2}{s^2 + \sigma_0^2}\in(0, 1)\quad (\sigma_0 \approx 8.0\mathrm{px}\ \mathrm{is\ the\ characteristic\ microscopic\ scale})$",
            ha="center", fontsize=7.2, color="#7C3AED", zorder=4)

    # ==========================================================================
    # 7. SLEEK PIPELINE DATAFLOW ARROWS
    # ==========================================================================
    draw_styled_arrow(ax, (14.0, 63.0), (16.5, 63.0), color="#0284C7", lw=1.6, zorder=8)
    draw_styled_arrow(ax, (31.0, 63.0), (33.5, 63.0), color=C_PRED_BLUE, lw=1.6, zorder=8)
    draw_styled_arrow(ax, (51.5, 63.0), (54.0, 63.0), color=C_FLOW_PURPLE, lw=1.6, zorder=8)
    ax.text(52.75, 65.5, "RoIs", ha="center", fontsize=7.0, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)
    draw_styled_arrow(ax, (78.5, 63.0), (81.0, 63.0), color=C_AMBER_ACCENT, lw=1.6, zorder=8)

    # Homotopy Core Feeding Arrows
    draw_styled_arrow(ax, (42.5, 26.0), (42.5, 32.0), color=C_FLOW_PURPLE, lw=1.4, dashed=True, zorder=8)
    ax.text(44.2, 29.0, "HLA Metric", fontsize=6.4, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)

    draw_styled_arrow(ax, (66.2, 26.0), (66.2, 32.0), color=C_FLOW_PURPLE, lw=1.4, dashed=True, zorder=8)
    ax.text(68.5, 29.0, "Loss Guidance", fontsize=6.4, fontweight="bold", color=C_FLOW_PURPLE, zorder=9)

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

    print(f"[SUCCESS] Pure Architecture Figure (Zero Score Contamination) successfully created:\n  - PDF: {out_pdf}\n  - PNG: {out_png}\n  - SVG: {out_svg}")


if __name__ == "__main__":
    render_pure_minimalist_fig5()
