"""
Ultra-Clean, Precision-Aligned Modular Block Architecture Diagram for H-WIoU.
Addresses all line shifts, text overlaps, and typography alignments:
- Perfectly calculated vertical rhythm (zero text collision)
- High-fidelity Loupe with clean target box separation
- Exact LaTeX mathtext sizing and padding
- Crisp inter-stage routing badges
"""
from __future__ import annotations
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Ellipse
from pathlib import Path
import shutil

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
FIG_DIR = ROOT / "journal/figures"
FIG_MANUSCRIPT_DIR = ROOT / "journal/manuscript/figures"
DATA_IMG_DIR = ROOT / "data/valid/images"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# Publication Typography Settings
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 300,
})


def draw_block(
    ax, x, y, w, h,
    title="", subtitle="", math_formula="",
    bg_color="#F8FAFC", border_color="#94A3B8", title_color="#0F172A",
    lw=1.2, radius=1.0, title_size=7.4, math_size=7.0, sub_size=6.0
):
    """Draw a clean, perfectly aligned modular block with dynamic vertical spacing."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=2
    )
    ax.add_patch(box)

    center_x = x + w / 2.0
    if title and math_formula and subtitle:
        ax.text(center_x, y + h * 0.78, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        ax.text(center_x, y + h * 0.48, math_formula, ha="center", va="center", fontsize=math_size, fontweight="bold", color="#1E293B", zorder=3)
        ax.text(center_x, y + h * 0.18, subtitle, ha="center", va="center", fontsize=sub_size, color="#64748B", zorder=3)
    elif title and math_formula:
        ax.text(center_x, y + h * 0.72, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        ax.text(center_x, y + h * 0.28, math_formula, ha="center", va="center", fontsize=math_size, fontweight="bold", color="#1E293B", zorder=3)
    elif title and subtitle:
        ax.text(center_x, y + h * 0.72, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)
        ax.text(center_x, y + h * 0.28, subtitle, ha="center", va="center", fontsize=sub_size, color="#475569", zorder=3)
    elif title:
        ax.text(center_x, y + h / 2.0, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=title_color, zorder=3)


def draw_container(ax, x, y, w, h, title="", title_bg="#1E3A8A", bg_color="#FFFFFF", border_color="#CBD5E1", radius=1.2, lw=1.2):
    """Draw a container lane for a pipeline stage."""
    card = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=1
    )
    ax.add_patch(card)

    if title:
        badge_h = 3.6
        badge_w = min(w - 2.0, len(title) * 0.66 + 5.0)
        badge_x = x + (w - badge_w) / 2.0
        badge_y = y + h - badge_h / 1.2
        pill = patches.FancyBboxPatch(
            (badge_x, badge_y), badge_w, badge_h,
            boxstyle=f"round,pad=0.0,rounding_size={radius*0.6}",
            facecolor=title_bg,
            edgecolor="none",
            zorder=3
        )
        ax.add_patch(pill)
        ax.text(
            badge_x + badge_w / 2.0, badge_y + badge_h / 2.0,
            title,
            ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="#FFFFFF",
            zorder=4
        )


def draw_arrow(ax, start, end, color="#2563EB", lw=1.5, rad=0.0, dashed=False, mutation=10):
    """Draw a clean, crisp directional arrow."""
    linestyle = "--" if dashed else "-"
    arrow = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=mutation,
        color=color,
        linewidth=lw,
        linestyle=linestyle,
        zorder=5
    )
    ax.add_patch(arrow)


def render_modular_architecture_diagram():
    """Render the ultra-clean, perfectly aligned architecture diagram."""
    fig, ax = plt.subplots(figsize=(18.0, 9.4), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Load authentic aerial image crop if available
    aerial_img = None
    if DATA_IMG_DIR.exists():
        candidates = sorted(list(DATA_IMG_DIR.glob("*.jpg")))
        if candidates:
            try:
                im = Image.open(candidates[0]).convert("RGB")
                aerial_img = np.array(im.crop((120, 100, 480, 370)).resize((400, 300)))
            except Exception:
                aerial_img = None

    # =========================================================================
    # TOP CONTAINER: Theoretical Scale-Homotopy Foundation (y in [64, 97])
    # =========================================================================
    draw_container(
        ax, x=1.5, y=64.0, w=97.0, h=33.0,
        title="Theoretical Foundation: Scale-Adaptive Homotopy Deformation Engine",
        title_bg="#5B21B6", bg_color="#FAF5FF", border_color="#DDD6FE"
    )

    # Block 1: Homotopy Metric Equation
    draw_block(
        ax, x=3.5, y=66.2, w=30.0, h=25.0,
        title="Unified Homotopy Metric",
        math_formula=r"$\mathcal{S}_{\mathrm{H\text{-}WIoU}} = [\mathrm{IoU}]^{\gamma(s)} \cdot \exp\left(-(1-\gamma(s))\mathcal{D}_{\mathcal{W}}^2\right)$",
        subtitle=r"$\ln \mathcal{S} = \gamma(s)\ln \mathrm{IoU} - (1-\gamma(s))\mathcal{D}_{\mathcal{W}}^2$",
        bg_color="#FFFFFF", border_color="#C4B5FD", title_color="#5B21B6",
        title_size=7.6, math_size=7.2, sub_size=6.6
    )

    # Block 2: Inset Scale Transition Curve
    draw_block(
        ax, x=35.0, y=66.2, w=30.0, h=25.0,
        title="Continuous Scale Transition Function",
        bg_color="#FFFFFF", border_color="#C4B5FD", title_color="#5B21B6",
        title_size=7.6
    )
    # Plot curve inside Block 2 with ample bottom and side margins
    sub_ax = ax.inset_axes([37.5, 68.8, 25.0, 16.5], transform=ax.transData)
    s_vals = np.linspace(0.1, 32, 100)
    sigma_0 = 8.0
    gamma_vals = (s_vals**2) / (s_vals**2 + sigma_0**2)
    sub_ax.plot(s_vals, gamma_vals, color="#7C3AED", lw=2.0, label=r"$\gamma(s) = \frac{s^2}{s^2+\sigma_0^2}$")
    sub_ax.axvline(x=8.0, color="#DC2626", linestyle=":", lw=1.2, label=r"$\sigma_0=8\mathrm{px}$")
    sub_ax.fill_between(s_vals[s_vals <= 8.0], 0, gamma_vals[s_vals <= 8.0], color="#EDE9FE", alpha=0.6)
    sub_ax.fill_between(s_vals[s_vals >= 8.0], 0, gamma_vals[s_vals >= 8.0], color="#DBEAFE", alpha=0.5)
    sub_ax.set_xlim(0, 32)
    sub_ax.set_ylim(0, 1.05)
    sub_ax.set_xlabel(r"Target Scale $s = \sqrt{w \cdot h}$ (px)", fontsize=5.4, labelpad=1.5)
    sub_ax.set_ylabel(r"Weight $\gamma(s)$", fontsize=5.4, labelpad=1.5)
    sub_ax.tick_params(labelsize=4.8, pad=1.5)
    sub_ax.grid(True, linestyle="--", alpha=0.35)
    sub_ax.legend(fontsize=5.2, loc="lower right", framealpha=0.92)

    # Block 3: Dual Asymptotic Regimes
    box3 = patches.FancyBboxPatch((66.5, 66.2), 30.0, 25.0, boxstyle="round,pad=0.0,rounding_size=1.0", facecolor="#FFFFFF", edgecolor="#C4B5FD", lw=1.2, zorder=2)
    ax.add_patch(box3)
    ax.text(81.5, 88.0, "Asymptotic Dual Regimes & Guarantees", ha="center", va="center", fontsize=7.6, fontweight="bold", color="#5B21B6", zorder=3)
    ax.text(81.5, 82.8, r"$\mathbf{1.\ Microscopic\ Regime\ (s < 8\mathrm{px}):}\ \gamma(s) \to 0$", ha="center", va="center", fontsize=6.6, fontweight="bold", color="#7C3AED", zorder=3)
    ax.text(81.5, 78.0, r"$\rightarrow\ \mathcal{S} \to \exp(-\mathcal{D}_{\mathcal{W}}^2)\ \Rightarrow\ \|\nabla_{\theta}\mathcal{L}\| = \mathcal{O}(1) > 0\ (\text{No Gradient Vanishing})$",
            ha="center", va="center", fontsize=6.0, color="#6D28D9", zorder=3)
    ax.text(81.5, 72.8, r"$\mathbf{2.\ Normal\ Scale\ Regime\ (s \gg \sigma_0):}\ \gamma(s) \to 1$", ha="center", va="center", fontsize=6.6, fontweight="bold", color="#2563EB", zorder=3)
    ax.text(81.5, 68.2, r"$\rightarrow\ \mathcal{S} \to \mathrm{IoU}\ \Rightarrow\ \text{Strict Lebesgue boundary alignment is fully preserved}$",
            ha="center", va="center", fontsize=6.0, color="#1D4ED8", zorder=3)

    # =========================================================================
    # BOTTOM CONTAINER 1: Multi-Scale Feature Extraction with Loupe (y in [3, 56])
    # =========================================================================
    draw_container(
        ax, x=1.5, y=3.0, w=30.5, h=53.0,
        title="1. Multi-Scale Feature Extraction",
        title_bg="#1E3A8A", bg_color="#F0F9FF", border_color="#BAE6FD"
    )

    # Block 1: Aerial Canvas + Circular Zoom Loupe
    canvas_card = patches.FancyBboxPatch(
        (3.2, 34.5), 27.1, 16.5,
        boxstyle="round,pad=0.0,rounding_size=1.0",
        facecolor="#FFFFFF", edgecolor="#93C5FD", lw=1.2, zorder=2
    )
    ax.add_patch(canvas_card)
    ax.text(16.75, 48.8, r"$\mathbf{Input\ Image\ }\mathbf{I} \in \mathbb{R}^{H \times W \times 3}\ \mathbf{with\ Micro\ Loupe}$", ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1E3A8A", zorder=3)

    # Mini Aerial Canvas Image
    img_x, img_y, img_w, img_h = 4.5, 35.8, 12.0, 10.5
    if aerial_img is not None:
        ax.imshow(aerial_img, extent=[img_x, img_x + img_w, img_y, img_y + img_h], aspect="auto", zorder=3)
        ax.add_patch(Rectangle((img_x, img_y), img_w, img_h, facecolor="none", edgecolor="#38BDF8", linewidth=1.0, zorder=4))
    else:
        ax.add_patch(Rectangle((img_x, img_y), img_w, img_h, facecolor="#0284C7", edgecolor="#38BDF8", linewidth=1.0, zorder=3))

    # Micro Target on Canvas
    tx, ty = 8.2, 40.5
    ax.add_patch(Rectangle((tx, ty), 1.0, 1.0, facecolor="none", edgecolor="#EF4444", linewidth=1.4, zorder=5))

    # Circular Loupe with Clean Vertical Spacing
    loupe_cx, loupe_cy, loupe_r = 22.5, 41.0, 3.5
    ax.plot([tx + 1.0, loupe_cx - loupe_r], [ty + 1.0, loupe_cy + 1.2], color="#EF4444", linestyle=":", lw=1.1, zorder=5)
    ax.plot([tx + 1.0, loupe_cx - loupe_r], [ty, loupe_cy - 1.2], color="#EF4444", linestyle=":", lw=1.1, zorder=5)
    ax.add_patch(Circle((loupe_cx, loupe_cy), loupe_r, facecolor="#0F172A", edgecolor="#EF4444", lw=1.5, zorder=6))
    
    # Target Box inside Loupe
    ax.add_patch(Rectangle((loupe_cx - 1.0, loupe_cy - 0.2), 2.0, 2.0, facecolor="#EF4444", alpha=0.5, edgecolor="#EF4444", lw=1.3, zorder=7))
    ax.text(loupe_cx, loupe_cy - 1.8, r"$\mathbf{4\times 4\ px\ Target}$", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#FCA5A5", zorder=8)
    ax.text(loupe_cx, loupe_cy + 2.4, r"$(s < 8\mathrm{px})$", ha="center", va="center", fontsize=5.2, color="#FCA5A5", zorder=8)

    # Block 2: ResNet-50 Feature Backbone
    draw_block(
        ax, x=3.2, y=20.0, w=27.1, h=11.5,
        title="ResNet-50 Feature Backbone",
        math_formula=r"$\{C_2, C_3, C_4, C_5\}$",
        subtitle="Bottom-up Feedforward Residual Stages",
        bg_color="#FFFFFF", border_color="#93C5FD", title_color="#1E3A8A",
        title_size=7.4, math_size=7.0, sub_size=6.0
    )

    # Block 3: Feature Pyramid Network
    draw_block(
        ax, x=3.2, y=5.5, w=27.1, h=11.5,
        title="Feature Pyramid Network (FPN)",
        math_formula=r"$\{P_2, P_3, P_4, P_5\}$",
        subtitle=r"$P_2\ (\text{stride 4}) \to \text{Micro},\quad P_3 \to \text{Tiny},\quad P_4 \to \text{Small},\quad P_5 \to \text{Normal}$",
        bg_color="#FFFFFF", border_color="#93C5FD", title_color="#1E3A8A",
        title_size=7.4, math_size=7.0, sub_size=5.6
    )

    draw_arrow(ax, (16.75, 34.5), (16.75, 31.8), color="#2563EB", lw=1.5)
    draw_arrow(ax, (16.75, 20.0), (16.75, 17.3), color="#2563EB", lw=1.5)

    # =========================================================================
    # BOTTOM CONTAINER 2: Stage 1 RPN Homotopy HLA (y in [3, 56])
    # =========================================================================
    draw_container(
        ax, x=33.5, y=3.0, w=32.0, h=53.0,
        title="2. Stage 1: RPN Homotopy Label Assignment",
        title_bg="#B45309", bg_color="#FFFBEB", border_color="#FDE68A"
    )

    draw_block(
        ax, x=35.2, y=37.5, w=28.6, h=13.0,
        title="Dense Anchor Generation",
        math_formula=r"$\mathcal{A} = \{A_i\}_{i=1}^N$",
        subtitle="Multi-scale anchor boxes across FPN levels P2-P5",
        bg_color="#FFFFFF", border_color="#FCD34D", title_color="#B45309",
        title_size=7.4, math_size=7.2, sub_size=6.0
    )

    draw_block(
        ax, x=35.2, y=21.5, w=28.6, h=13.0,
        title="Homotopy Label Assignment (HLA)",
        math_formula=r"$\mathbf{S}_{ij} = \mathcal{S}_{\mathrm{H\text{-}WIoU}}(A_i, G_j)$",
        subtitle=r"$\text{Continuous Gaussian Transport } \mathcal{W}_2 \text{ replaces brittle IoU}$",
        bg_color="#FEF3C7", border_color="#F59E0B", title_color="#92400E", lw=1.5,
        title_size=7.6, math_size=7.4, sub_size=6.0
    )

    draw_block(
        ax, x=35.2, y=5.5, w=28.6, h=13.0,
        title="Region Proposal Generation",
        math_formula=r"$\mathbf{R} = \{\mathbf{r}_k\}_{k=1}^K$",
        subtitle=r"$\text{Positive Threshold } t_{\mathrm{pos}}=0.7 \Rightarrow \text{High-Quality RoI Candidates}$",
        bg_color="#FFFFFF", border_color="#FCD34D", title_color="#B45309",
        title_size=7.4, math_size=7.2, sub_size=6.0
    )

    draw_arrow(ax, (49.5, 37.5), (49.5, 34.8), color="#D97706", lw=1.5)
    draw_arrow(ax, (49.5, 21.5), (49.5, 18.8), color="#D97706", lw=1.5)

    # =========================================================================
    # BOTTOM CONTAINER 3: Stage 2 RoI Head & Multi-Task Loss (y in [3, 56])
    # =========================================================================
    draw_container(
        ax, x=67.0, y=3.0, w=31.5, h=53.0,
        title="3. Stage 2: RoI Head & Multi-Task Loss",
        title_bg="#065F46", bg_color="#F0FDF4", border_color="#A7F3D0"
    )

    draw_block(
        ax, x=68.5, y=41.5, w=28.5, h=9.8,
        title="RoIAlign Feature Pooling",
        math_formula=r"$7 \times 7\ \text{Bilinear Spatial Sampling per RoI}$",
        subtitle="Extracts fixed-length feature maps from FPN",
        bg_color="#FFFFFF", border_color="#86EFAC", title_color="#065F46",
        title_size=7.2, math_size=6.6, sub_size=5.8
    )

    draw_block(
        ax, x=68.5, y=29.0, w=28.5, h=9.8,
        title="Two-Layer MLP Head",
        math_formula=r"$2 \times 1024\text{-d Fully-Connected Layers}$",
        subtitle="Shared representation for classification and regression",
        bg_color="#FFFFFF", border_color="#86EFAC", title_color="#065F46",
        title_size=7.2, math_size=6.6, sub_size=5.8
    )

    draw_block(
        ax, x=68.5, y=16.5, w=28.5, h=10.0,
        title="Multi-Task Loss Supervision",
        math_formula=r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{cls}}(\hat{\mathbf{p}}, y) + \lambda \cdot [1 - \mathcal{S}_{\mathrm{H\text{-}WIoU}}(\hat{\mathbf{b}}, \mathbf{g})]$",
        subtitle=r"$\mathcal{L}_{\mathrm{H\text{-}WIoU}} \in [0, 1] \text{ ensures strictly bounded box regression}$",
        bg_color="#DCFCE7", border_color="#22C55E", title_color="#064E3B", lw=1.5,
        title_size=7.4, math_size=6.5, sub_size=5.8
    )

    draw_block(
        ax, x=68.5, y=5.5, w=28.5, h=8.5,
        title="Final Detection Instances",
        math_formula=r"$\mathcal{Y} = \{(\hat{y}_k, \hat{\mathbf{b}}_k, \hat{s}_k)\}_{k=1}^M$",
        subtitle="Post-processed via Class-Aware NMS",
        bg_color="#FFFFFF", border_color="#86EFAC", title_color="#065F46",
        title_size=7.2, math_size=6.6, sub_size=5.8
    )

    draw_arrow(ax, (82.75, 41.5), (82.75, 39.0), color="#059669", lw=1.5)
    draw_arrow(ax, (82.75, 29.0), (82.75, 26.8), color="#059669", lw=1.5)
    draw_arrow(ax, (82.75, 16.5), (82.75, 14.3), color="#059669", lw=1.5)

    # =========================================================================
    # PIPELINE DATAFLOW & CONTROL ARROWS (Clean, Non-Colliding Routes)
    # =========================================================================
    # 1. FPN -> RPN Anchor Gen (Solid Blue)
    draw_arrow(ax, (30.3, 11.25), (35.2, 44.0), color="#2563EB", lw=1.6, rad=0.10)
    badge1 = patches.FancyBboxPatch((30.4, 26.2), 4.7, 2.8, boxstyle="round,pad=0.15", facecolor="#EFF6FF", edgecolor="#3B82F6", lw=0.8, zorder=6)
    ax.add_patch(badge1)
    ax.text(32.75, 27.6, "FPN Features", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#1D4ED8", zorder=7)

    # 2. RPN Proposals -> RoIAlign (Solid Amber)
    draw_arrow(ax, (63.8, 12.0), (68.5, 46.4), color="#D97706", lw=1.6, rad=-0.10)
    badge2 = patches.FancyBboxPatch((63.8, 27.8), 4.7, 2.8, boxstyle="round,pad=0.15", facecolor="#FFFBEB", edgecolor="#F59E0B", lw=0.8, zorder=6)
    ax.add_patch(badge2)
    ax.text(66.15, 29.2, "Candidate RoIs", ha="center", va="center", fontsize=5.8, fontweight="bold", color="#B45309", zorder=7)

    # 3. Top Homotopy Core -> Stage 1 HLA (Vertical Dashed Purple)
    draw_arrow(ax, (49.5, 64.0), (49.5, 54.0), color="#7C3AED", lw=1.6, dashed=True)
    ax.text(49.5, 59.0, r"$\gamma(s)\ \text{Modulation for HLA}$", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#5B21B6", bbox=dict(boxstyle="round,pad=0.22", facecolor="#EDE9FE", edgecolor="#8B5CF6", lw=0.8), zorder=6)

    # 4. Top Homotopy Core -> Stage 2 Box Loss (Vertical Dashed Purple)
    draw_arrow(ax, (82.75, 64.0), (82.75, 54.0), color="#7C3AED", lw=1.6, dashed=True)
    ax.text(82.75, 59.0, r"$\gamma(s)\ \text{Modulation for Box Loss}$", ha="center", va="center", fontsize=6.5, fontweight="bold", color="#5B21B6", bbox=dict(boxstyle="round,pad=0.22", facecolor="#EDE9FE", edgecolor="#8B5CF6", lw=0.8), zorder=6)

    out_pdf = FIG_DIR / "fig5_pipeline_architecture.pdf"
    out_png = FIG_DIR / "fig5_pipeline_architecture.png"
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    plt.close()

    # Mirror to manuscript figures directory
    shutil.copy(out_pdf, FIG_MANUSCRIPT_DIR / out_pdf.name)
    shutil.copy(out_png, FIG_MANUSCRIPT_DIR / out_png.name)

    print(f"Precision-Aligned Modular Block Architecture Diagram generated successfully -> {out_pdf} and {out_png}")


if __name__ == "__main__":
    render_modular_architecture_diagram()
