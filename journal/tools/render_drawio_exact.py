"""
Exact Draw.io Vector Renderer for Publication Figures.
Translates draw.io layout into publication-grade vector PDF and 300 DPI PNG.
"""
from __future__ import annotations
import math
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Polygon

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
OUT_DIR = ROOT / "journal/manuscript/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})

def draw_drawio_box(ax, x, y, w, h, text="", subtext="", bg_color="#FFFFFF", border_color="#CBD5E1", 
                    text_color="#0F172A", font_size=8.5, radius=1.2, lw=1.2, zorder=2, bold=False):
    """Draws a clean draw.io-style rounded box with multi-line text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(box)
    
    if text and not subtext:
        ax.text(
            x + w / 2.0, y + h / 2.0,
            text,
            ha="center", va="center",
            fontsize=font_size, fontweight="bold" if bold else "normal", color=text_color,
            zorder=zorder + 1
        )
    elif text and subtext:
        ax.text(
            x + w / 2.0, y + h * 0.68,
            text,
            ha="center", va="center",
            fontsize=font_size, fontweight="bold" if bold else "normal", color=text_color,
            zorder=zorder + 1
        )
        ax.text(
            x + w / 2.0, y + h * 0.32,
            subtext,
            ha="center", va="center",
            fontsize=font_size * 0.85, color="#475569",
            zorder=zorder + 1
        )

def draw_arrow_conn(ax, x1, y1, x2, y2, color="#475569", lw=1.6, dashed=False, zorder=4):
    """Draws a smooth orthogonal connector arrow."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle="--" if dashed else "-",
            mutation_scale=11,
            shrinkA=1, shrinkB=2,
        ),
        zorder=zorder
    )

# ==============================================================================
# RENDER FIGURE 1 (Draw.io Homotopy Theory)
# ==============================================================================
def render_fig1_homotopy():
    print("Rendering Figure 1 from drawio specification...")
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 44)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Card 1: Standard IoU Collapse
    draw_drawio_box(ax, 3, 2, 30, 40, bg_color="#FEF2F2", border_color="#FECACA", radius=1.8, lw=1.5, zorder=1)
    draw_drawio_box(ax, 5, 36.5, 26, 4.2, text="(a) Standard IoU Collapse", bg_color="#991B1B", border_color="none", text_color="#FFFFFF", font_size=8.5, bold=True, zorder=2)
    
    # Anchor & GT boxes
    draw_drawio_box(ax, 6, 23, 9.5, 9.5, text="Anchor $A$\n$(8\\times 8\\mathrm{px})$", bg_color="#FCA5A5", border_color="#DC2626", text_color="#7F1D1D", font_size=7.2, bold=True, zorder=3)
    draw_drawio_box(ax, 20, 24.5, 8.5, 8.5, text="GT $G$\n$(6\\times 6\\mathrm{px})$", bg_color="#93C5FD", border_color="#2563EB", text_color="#1E3A8A", font_size=7.2, bold=True, zorder=3)
    
    # Description box
    draw_drawio_box(ax, 5, 4.5, 26, 16.5, bg_color="#FFFFFF", border_color="#FECACA", radius=1.2, lw=1.0, zorder=2)
    ax.text(18, 17.5, r"$\mathrm{Area}(A \cap G) = 0$", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#991B1B", zorder=3)
    ax.text(18, 13.0, r"$\mathrm{IoU}(A, G) = 0$", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#991B1B", zorder=3)
    ax.text(18, 9.5, r"$\nabla_A \mathrm{IoU} = \mathbf{0} \quad (\text{Vanishing})$", ha="center", va="center", fontsize=8.0, color="#7F1D1D", zorder=3)
    ax.text(18, 6.0, "Zero feedback on sub-pixel shifts", ha="center", va="center", fontsize=6.8, style="italic", color="#B91C1C", zorder=3)

    # Connector 1 -> 2
    draw_arrow_conn(ax, 33.5, 22, 36.5, 22, color="#64748B", lw=2.0)

    # Card 2: Gaussian Wasserstein Space
    draw_drawio_box(ax, 37, 2, 31, 40, bg_color="#F0FDF4", border_color="#BBF7D0", radius=1.8, lw=1.5, zorder=1)
    draw_drawio_box(ax, 39, 36.5, 27, 4.2, text="(b) Gaussian Wasserstein Space", bg_color="#166534", border_color="none", text_color="#FFFFFF", font_size=8.5, bold=True, zorder=2)
    
    # Gaussian Ellipses
    e1 = Ellipse((45, 27), 9.0, 7.0, angle=25, facecolor="#86EFAC", edgecolor="#16A34A", lw=1.5, alpha=0.7, zorder=3)
    ax.add_patch(e1)
    ax.text(45, 27, r"$\mathcal{N}_A(\mu_A, \Sigma_A)$", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#14532D", zorder=4)

    e2 = Ellipse((59, 29), 7.5, 5.5, angle=-15, facecolor="#93C5FD", edgecolor="#2563EB", lw=1.5, alpha=0.7, zorder=3)
    ax.add_patch(e2)
    ax.text(59, 29, r"$\mathcal{N}_G(\mu_G, \Sigma_G)$", ha="center", va="center", fontsize=6.8, fontweight="bold", color="#1E3A8A", zorder=4)

    draw_arrow_conn(ax, 49.5, 27.5, 54.5, 28.5, color="#16A34A", lw=1.8, dashed=True)
    ax.text(52, 30.5, r"$W_2(A, G)$", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#15803D", zorder=5)

    # Description box
    draw_drawio_box(ax, 39, 4.5, 27, 16.5, bg_color="#FFFFFF", border_color="#BBF7D0", radius=1.2, lw=1.0, zorder=2)
    ax.text(52.5, 17.5, r"Optimal Transport Distance", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#166534", zorder=3)
    ax.text(52.5, 13.5, r"$W_2^2 = \|\mu_A - \mu_G\|_2^2 + \dots$", ha="center", va="center", fontsize=7.8, color="#15803D", zorder=3)
    ax.text(52.5, 9.5, r"$\mathrm{NWD} = \exp\left(-\frac{W_2}{C}\right) > 0$", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#14532D", zorder=3)
    ax.text(52.5, 6.0, "Smooth metric across all $\mathbb{R}^2$", ha="center", va="center", fontsize=6.8, style="italic", color="#166534", zorder=3)

    # Connector 2 -> 3
    draw_arrow_conn(ax, 68.5, 22, 71.5, 22, color="#64748B", lw=2.0)

    # Card 3: Continuous Homotopy H-WIoU (Proposed)
    draw_drawio_box(ax, 72, 2, 31, 40, bg_color="#FFF7ED", border_color="#EA580C", radius=1.8, lw=2.0, zorder=1)
    draw_drawio_box(ax, 74, 36.5, 27, 4.2, text="(c) Continuous Homotopy H-WIoU", bg_color="#EA580C", border_color="none", text_color="#FFFFFF", font_size=8.5, bold=True, zorder=2)
    
    # Formula box
    draw_drawio_box(ax, 74, 21.5, 27, 13.5, bg_color="#FFFFFF", border_color="#FDBA74", radius=1.2, lw=1.2, zorder=2)
    ax.text(87.5, 30.5, r"$H_\gamma(A, G) = (1 - \gamma)\mathrm{IoU} + \gamma \mathrm{NWD}$", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#C2410C", zorder=3)
    ax.text(87.5, 25.5, r"$\gamma(d) = \frac{1}{1 + (d / \sigma_0)^2}, \quad d = \sqrt{w \cdot h}$", ha="center", va="center", fontsize=7.4, color="#9A3412", zorder=3)

    # Regimes box
    draw_drawio_box(ax, 74, 4.5, 27, 15.5, bg_color="#FFEDD5", border_color="#FDBA74", radius=1.2, lw=1.0, zorder=2)
    ax.text(87.5, 16.5, r"$\bullet \ d \to 0 \ (\text{Tiny}): \ \gamma \to 1 \Rightarrow H_\gamma \to \mathrm{NWD}$", ha="center", va="center", fontsize=7.2, color="#9A3412", zorder=3)
    ax.text(87.5, 12.5, r"$\bullet \ d \to \infty \ (\text{Normal}): \ \gamma \to 0 \Rightarrow H_\gamma \to \mathrm{IoU}$", ha="center", va="center", fontsize=7.2, color="#9A3412", zorder=3)
    ax.text(87.5, 7.5, "Strict Smoothness & Non-Zero Gradient", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#C2410C", zorder=3)

    plt.tight_layout()
    pdf_out = OUT_DIR / "fig1_homotopy_theory.pdf"
    png_out = OUT_DIR / "fig1_homotopy_theory.png"
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Figure 1 to {pdf_out} and {png_out}")


# ==============================================================================
# RENDER FIGURE 5 (Draw.io Pipeline Architecture)
# ==============================================================================
def render_fig5_pipeline():
    print("Rendering Figure 5 from drawio specification...")
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    ax.set_xlim(0, 115)
    ax.set_ylim(0, 52)
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")

    # Banner Header
    draw_drawio_box(ax, 2, 46.5, 111, 4.5, text="H-WIoU: End-to-End Scale-Aware Homotopy Detection Pipeline for Tiny Objects", bg_color="#F8FAFC", border_color="#E2E8F0", text_color="#0F172A", font_size=9.2, bold=True, zorder=1)

    # Container 1: Multi-Scale Backbone
    draw_drawio_box(ax, 2, 2, 27, 43, bg_color="#F8FAFC", border_color="#CBD5E1", radius=1.8, lw=1.5, zorder=1)
    draw_drawio_box(ax, 3.5, 40.5, 24, 3.5, text="Stage 1: Multi-Scale Backbone", bg_color="#1E293B", border_color="none", text_color="#FFFFFF", font_size=8.0, bold=True, zorder=2)
    
    draw_drawio_box(ax, 5, 31.5, 21, 6.5, text="Input Aerial Image", subtext="1024 × 1024 px (Tiny < 16px)", bg_color="#FFFFFF", border_color="#94A3B8", font_size=7.5, zorder=2)
    draw_drawio_box(ax, 5, 23.0, 21, 6.0, text="ResNet-50 Backbone", subtext="C2, C3, C4, C5 Stages", bg_color="#EFF6FF", border_color="#3B82F6", font_size=7.5, zorder=2)
    
    fpn_levels = [("FPN P2 (Stride 4)", 18.0, "#DBEAFE"), ("FPN P3 (Stride 8)", 14.0, "#BFDBFE"), ("FPN P4 (Stride 16)", 10.0, "#93C5FD"), ("FPN P5 (Stride 32)", 6.0, "#60A5FA")]
    for lbl, ypos, col in fpn_levels:
        draw_drawio_box(ax, 5, ypos, 21, 3.3, text=lbl, bg_color=col, border_color="#2563EB", font_size=6.8, bold=True, zorder=2)
    
    draw_arrow_conn(ax, 15.5, 31.5, 15.5, 29.0)
    draw_arrow_conn(ax, 15.5, 23.0, 15.5, 21.3)

    # Connector 1 -> 2
    draw_arrow_conn(ax, 29.5, 23.5, 32.5, 23.5, color="#2563EB", lw=2.0)

    # Container 2: Dynamic Homotopy RPN Assignment (Proposed)
    draw_drawio_box(ax, 33, 2, 42, 43, bg_color="#FFF7ED", border_color="#EA580C", radius=1.8, lw=2.0, zorder=1)
    draw_drawio_box(ax, 34.5, 40.5, 39, 3.5, text="Stage 2: Homotopy Soft-Assignment (RPN) [PROPOSED]", bg_color="#EA580C", border_color="none", text_color="#FFFFFF", font_size=8.0, bold=True, zorder=2)

    # Dynamic Weighting
    draw_drawio_box(ax, 35.5, 31.0, 37, 8.0, bg_color="#FFFFFF", border_color="#FB923C", radius=1.2, lw=1.2, zorder=2)
    ax.text(54, 35.8, r"1. Dynamic Scale Parameter $\gamma(d)$:", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#9A3412", zorder=3)
    ax.text(54, 32.5, r"$\gamma(d) = \frac{1}{1 + (d / \sigma_0)^2}, \quad \sigma_0 = 8.0\mathrm{px}$", ha="center", va="center", fontsize=7.8, color="#C2410C", zorder=3)

    # Continuous Combination
    draw_drawio_box(ax, 35.5, 21.5, 37, 8.0, bg_color="#FFFFFF", border_color="#FB923C", radius=1.2, lw=1.2, zorder=2)
    ax.text(54, 26.3, r"2. Continuous Homotopy Metric Space:", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#9A3412", zorder=3)
    ax.text(54, 23.0, r"$H_\gamma(A, G) = (1 - \gamma)\mathrm{IoU} + \gamma \mathrm{NWD}$", ha="center", va="center", fontsize=7.8, color="#C2410C", zorder=3)

    # Regimes Box
    draw_drawio_box(ax, 35.5, 13.5, 37, 6.5, bg_color="#FFEDD5", border_color="#FDBA74", radius=1.0, lw=1.0, zorder=2)
    ax.text(54, 17.5, r"$\bullet \ d < 8\mathrm{px}: \gamma \to 1 \Rightarrow H_\gamma \to \mathrm{NWD}$ (No IoU Collapse)", ha="center", va="center", fontsize=6.8, color="#9A3412", zorder=3)
    ax.text(54, 15.0, r"$\bullet \ d > 32\mathrm{px}: \gamma \to 0 \Rightarrow H_\gamma \to \mathrm{IoU}$ (Standard Metric)", ha="center", va="center", fontsize=6.8, color="#9A3412", zorder=3)

    # Soft positive assignment output
    draw_drawio_box(ax, 35.5, 4.5, 37, 7.5, text="3. Soft-Label Positive Candidates", subtext="Top-k proposals with non-zero gradients across scales", bg_color="#FED7AA", border_color="#EA580C", text_color="#7C2D12", font_size=7.4, bold=True, zorder=2)

    draw_arrow_conn(ax, 54, 31.0, 54, 29.5, color="#EA580C", lw=1.5)
    draw_arrow_conn(ax, 54, 21.5, 54, 20.0, color="#EA580C", lw=1.5)
    draw_arrow_conn(ax, 54, 13.5, 54, 12.0, color="#EA580C", lw=1.5)

    # Connector 2 -> 3
    draw_arrow_conn(ax, 75.5, 23.5, 78.5, 23.5, color="#16A34A", lw=2.0)

    # Container 3: Fast R-CNN Head & Bounded Loss
    draw_drawio_box(ax, 79, 2, 34, 43, bg_color="#F0FDF4", border_color="#16A34A", radius=1.8, lw=1.5, zorder=1)
    draw_drawio_box(ax, 80.5, 40.5, 31, 3.5, text="Stage 3: Fast R-CNN & Bounded Loss", bg_color="#166534", border_color="none", text_color="#FFFFFF", font_size=8.0, bold=True, zorder=2)

    draw_drawio_box(ax, 81.5, 32.0, 29, 6.8, text="RoIAlign Feature Pooling", subtext="7 × 7 Bilinear Interpolation + 2×FC(1024)", bg_color="#FFFFFF", border_color="#4ADE80", font_size=7.2, zorder=2)
    
    draw_drawio_box(ax, 81.5, 23.5, 29, 6.8, text="Classification Branch", subtext=r"$\mathcal{L}_{\mathrm{cls}} = \mathrm{CrossEntropy}(p, y)$", bg_color="#DCFCE7", border_color="#22C55E", font_size=7.2, zorder=2)

    draw_drawio_box(ax, 81.5, 12.5, 29, 9.2, bg_color="#FFFFFF", border_color="#16A34A", radius=1.2, lw=1.5, zorder=2)
    ax.text(96, 18.5, r"Bounded Homotopy Regression:", ha="center", va="center", fontsize=7.2, fontweight="bold", color="#15803D", zorder=3)
    ax.text(96, 15.0, r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - H_\gamma(B_{\mathrm{pred}}, B_{\mathrm{gt}})$", ha="center", va="center", fontsize=7.8, color="#166534", zorder=3)

    draw_drawio_box(ax, 81.5, 4.5, 29, 6.5, text="Final Calibrated Detections", subtext=r"$\mathrm{mAP}_{50}: 46.2\% \ (+19.9\%) \ | \ \mathrm{Fair\text{-}20}: 46.34\%$", bg_color="#166534", border_color="none", text_color="#FFFFFF", font_size=7.2, bold=True, zorder=2)

    draw_arrow_conn(ax, 96, 32.0, 96, 30.3, color="#16A34A", lw=1.5)
    draw_arrow_conn(ax, 96, 23.5, 96, 21.7, color="#16A34A", lw=1.5)
    draw_arrow_conn(ax, 96, 12.5, 96, 11.0, color="#16A34A", lw=1.5)

    plt.tight_layout()
    pdf_out = OUT_DIR / "fig5_pipeline_architecture.pdf"
    png_out = OUT_DIR / "fig5_pipeline_architecture.png"
    plt.savefig(pdf_out, format="pdf", bbox_inches="tight")
    plt.savefig(png_out, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved Figure 5 to {pdf_out} and {png_out}")

def main():
    print("=" * 80)
    print("   EXACT DRAW.IO VECTOR RENDERING FOR IEEE TPAMI MANUSCRIPT")
    print("=" * 80)
    render_fig1_homotopy()
    render_fig5_pipeline()
    print("=" * 80)

if __name__ == "__main__":
    main()
