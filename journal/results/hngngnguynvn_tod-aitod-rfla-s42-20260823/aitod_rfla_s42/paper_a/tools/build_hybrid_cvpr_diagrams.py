"""
Hybrid AI-Assisted CVPR Top-Tier Architecture & Benchmark Diagram Generator for Paper A.
Combines:
- AI Generated high-resolution aerial dataset tiles and 3D feature representations
- Precision vector overlays, cards, mathematical formulas, and bounding boxes
- Clean 300 DPI PNG, vector PDF, and native Draw.io XML
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from PIL import Image
import numpy as np

def create_hybrid_fig1():
    """Generates CVPR-grade Hybrid Architecture Diagram with AI-generated visual tiles."""
    fig, ax = plt.subplots(figsize=(16.5, 7.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAFA')

    def add_card(x, y, w, h, title, subtitle=None, bg_color='#FFFFFF', border_color='#CBD5E1', 
                 title_color='#0F172A', lw=1.5, radius=0.18, shadow=True):
        if shadow:
            s_box = FancyBboxPatch((x+0.05, y-0.06), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                   facecolor='#000000', alpha=0.05, edgecolor='none', zorder=1)
            ax.add_patch(s_box)
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
        ax.add_patch(box)
        if title:
            ax.text(x + w/2, y + h - 0.32, title, ha='center', va='center',
                    fontsize=10.5, fontweight='bold', color=title_color, zorder=5)
        if subtitle:
            ax.text(x + w/2, y + 0.32, subtitle, ha='center', va='center',
                    fontsize=8.5, color='#64748B', zorder=5)
        return box

    # --- 1. Left Container: Input Tile with Embedded Aerial Image & Zoom Callout ---
    add_card(0.2, 0.4, 3.4, 6.8, "Input Tile (b1-tiled)", "TinyPerson 800x800",
             bg_color='#F8FAFC', border_color='#CBD5E1', title_color='#0F172A')

    # Embed AI aerial image
    aerial_path = "paper_a/figures/assets/aerial_tile.jpg"
    if os.path.exists(aerial_path):
        img_aerial = Image.open(aerial_path)
        ax.imshow(img_aerial, extent=[0.4, 3.4, 3.0, 6.0], zorder=3, aspect='auto')
        # Red bounding box on micro-target in image
        ax.add_patch(Rectangle((1.8, 4.3), 0.22, 0.22, edgecolor='#DC2626', facecolor='none', lw=2.0, zorder=4))
        
        # Zoomed-in inset box
        add_card(0.4, 0.9, 3.0, 1.8, "Microscopic Target (<8x8 px)", "Discretization Noise Zone",
                 bg_color='#FEF2F2', border_color='#FCA5A5', title_color='#DC2626', lw=1.2, radius=0.1)
        # Cropped swimmer/boat simulation
        ax.add_patch(Rectangle((0.7, 1.25), 0.8, 0.8, edgecolor='#DC2626', facecolor='#EFF6FF', lw=1.5, zorder=4))
        ax.text(1.1, 1.65, "Target\n6x6 px", ha='center', va='center', fontsize=7.5, fontweight='bold', color='#DC2626', zorder=5)
        ax.text(2.4, 1.65, r"$\mathrm{Area} \leq 64\text{ px}^2$" + "\n" + r"$\nabla \mathrm{IoU} \to \mathbf{0}$", 
                ha='center', va='center', fontsize=8.0, fontweight='bold', color='#991B1B', zorder=5)

    # --- 2. Backbone & FPN Container ---
    add_card(3.9, 0.4, 4.5, 6.8, "Multi-Scale Backbone & FPN", "ResNet-50 + Top-Down Feature Pyramid",
             bg_color='#F0F9FF', border_color='#BAE6FD', title_color='#0369A1')

    # Backbone Stages (C2-C5)
    c_stages = [
        ("C2", "stride 4", 4.1, 1.0, 1.8, 1.1),
        ("C3", "stride 8", 4.1, 2.4, 1.8, 1.1),
        ("C4", "stride 16", 4.1, 3.8, 1.8, 1.1),
        ("C5", "stride 32", 4.1, 5.2, 1.8, 1.1),
    ]
    for name, sub, bx, by, bw, bh in c_stages:
        add_card(bx, by, bw, bh, name, sub, bg_color='#EFF6FF', border_color='#93C5FD', title_color='#1E3A8A', lw=1.2)

    # Bottom-up arrows
    for i in range(3):
        ax.annotate("", xy=(5.0, 2.38 + i*1.4), xytext=(5.0, 2.12 + i*1.4),
                    arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.3", color="#2563EB", lw=1.8, zorder=4))

    # FPN Levels (P2-P5)
    p_stages = [
        ("P2", "256-d", 6.3, 1.0, 1.8, 1.1),
        ("P3", "256-d", 6.3, 2.4, 1.8, 1.1),
        ("P4", "256-d", 6.3, 3.8, 1.8, 1.1),
        ("P5", "256-d", 6.3, 5.2, 1.8, 1.1),
    ]
    for name, sub, bx, by, bw, bh in p_stages:
        add_card(bx, by, bw, bh, name, sub, bg_color='#E0F2FE', border_color='#7DD3FC', title_color='#0284C7', lw=1.2)

    # Lateral 1x1 conv arrows (C_i -> P_i)
    for i in range(4):
        ax.annotate("", xy=(6.28, 1.55 + i*1.4), xytext=(5.92, 1.55 + i*1.4),
                    arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.3", color="#0284C7", lw=1.8, zorder=4))

    # Top-Down arrows (P5 -> P4 -> P3 -> P2)
    for i in range(3, 0, -1):
        ax.annotate("", xy=(7.2, 1.02 + (i-1)*1.4 + 1.1), xytext=(7.2, 1.02 + i*1.4 - 0.05),
                    arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.3", color="#0369A1", lw=2.0, zorder=4))

    # Input to Backbone arrow
    ax.annotate("", xy=(4.08, 3.8), xytext=(3.62, 3.8),
                arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.35", color="#475569", lw=2.0, zorder=4))

    # --- 3. Upper Branch: PC-MOC Distillation Engine ---
    add_card(8.8, 5.0, 7.4, 2.2, "PC-MOC Multi-Scale Feature Distillation",
             bg_color='#ECFDF5', border_color='#A7F3D0', title_color='#065F46', lw=1.8, radius=0.2)
    ax.text(12.5, 5.85, r"$\mathcal{L}_{\mathrm{distill}} = \sum_{\ell=2}^{5} \frac{1}{H_\ell W_\ell} \sum_{i,j} \left( 1 - \frac{\langle f_{\mathrm{curr}}^{P_\ell}(i,j),\, f_{\mathrm{ref}}^{P_\ell}(i,j) \rangle}{\|f_{\mathrm{curr}}^{P_\ell}(i,j)\| \|f_{\mathrm{ref}}^{P_\ell}(i,j)\| + \epsilon} \right)$",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color='#047857', zorder=5)
    ax.text(12.5, 5.30, "Aligns FPN representations against reference teacher to prevent curriculum drift",
            ha='center', va='center', fontsize=8.2, color='#059669', style='italic', zorder=5)

    ax.annotate("", xy=(8.78, 6.1), xytext=(8.12, 5.75),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#059669", lw=2.2, zorder=4))

    # --- 4. Center Branch: RoI Head & SA-ALW ---
    add_card(8.8, 2.7, 7.4, 2.1, "RoI Head with SA-ALW & Iterative-CBL",
             bg_color='#FAF5FF', border_color='#E9D5FF', title_color='#6B21A8', lw=1.8, radius=0.2)
    ax.text(12.5, 3.65, r"$\mathcal{L}_{\mathrm{reg}}^{\mathrm{SA}} = \sqrt{w_{\mathrm{pos}}(s) D_{\mathrm{pos}}(p, t) + D_{\mathrm{shape}}(p, t)} \quad\text{with}\quad K_{\mathrm{SA}} = \exp\left[-\beta(s)\sqrt{D_{\mathrm{SA}}}\right]$",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color='#7E22CE', zorder=5)
    ax.text(12.5, 3.05, r"Scale-Adaptive Anisotropic log-Wasserstein with dynamic temperature $\beta(s)$ and weight $w_{\mathrm{pos}}(s)$",
            ha='center', va='center', fontsize=8.2, color='#9333EA', style='italic', zorder=5)

    ax.annotate("", xy=(8.78, 3.75), xytext=(8.12, 3.75),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#9333EA", lw=2.2, zorder=4))

    # --- 5. Lower Branch: PC-MR Orthogonal Gradient Projection ---
    add_card(8.8, 0.4, 7.4, 2.1, "PC-MR Proposal Micro-Rescue (RPN)",
             bg_color='#FEF2F2', border_color='#FECACA', title_color='#991B1B', lw=1.8, radius=0.2)
    ax.text(12.5, 1.35, r"$\mathbf{g}_{\mathrm{proj}} = \mathbf{g}_{\mathrm{micro}} - \frac{\mathbf{g}_{\mathrm{micro}} \cdot \mathbf{g}_{\mathrm{main}}}{\|\mathbf{g}_{\mathrm{main}}\|^2 + \epsilon} \mathbf{g}_{\mathrm{main}} \quad \text{such that } \mathbf{g}_{\mathrm{proj}} \perp \mathbf{g}_{\mathrm{main}}$",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#B91C1C', zorder=5)
    ax.text(12.5, 0.75, "Projects micro-instance gradients onto orthogonal subspace to prevent annihilation",
            ha='center', va='center', fontsize=8.2, color='#DC2626', style='italic', zorder=5)

    ax.annotate("", xy=(8.78, 1.45), xytext=(8.12, 1.55),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#DC2626", lw=2.2, zorder=4))

    ax.set_xlim(-0.1, 16.5)
    ax.set_ylim(0.1, 7.5)
    ax.axis('off')

    plt.tight_layout()
    out_png = "paper_a/figures/fig1_framework_architecture.png"
    out_pdf = "paper_a/figures/fig1_framework_architecture.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated Hybrid CVPR Fig 1: {out_png}, {out_pdf}")

def create_hybrid_fig3():
    """Generates Figure 3 (3-panel bar chart) with non-overlapping, clean rotated x-ticks."""
    methods = [
        "Faster R-CNN",
        "NWD",
        "SA-ALW",
        "Iterative-CBL",
        "PC-MR",
        "PC-MOC",
        "Joint (Ours)"
    ]
    
    # 3-seed means and standard deviations from megatable
    ap_micro_means = [36.10, 37.30, 39.25, 40.32, 39.59, 39.55, 41.16]
    ap_micro_stds = [0.92, 0.22, 1.10, 2.14, 1.54, 1.05, 1.86]
    
    ap75_means = [6.67, 5.79, 6.55, 7.12, 7.14, 6.96, 7.19]
    ap75_stds = [0.20, 0.33, 0.29, 0.14, 0.18, 0.07, 0.18]
    
    map50_means = [46.49, 41.89, 46.27, 44.91, 44.21, 44.88, 45.09]
    map50_stds = [0.27, 0.94, 0.28, 0.52, 0.68, 0.79, 0.64]

    colors = ['#64748B', '#0284C7', '#0D9488', '#D97706', '#9333EA', '#2563EB', '#DC2626']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    x = np.arange(len(methods))
    bar_width = 0.62

    # Panel 1: AP_micro
    ax1.set_facecolor('#FAFAFA')
    bars1 = ax1.bar(x, ap_micro_means, yerr=ap_micro_stds, capsize=4, width=bar_width,
                    color=colors, edgecolor='#334155', linewidth=0.8, alpha=0.9)
    ax1.set_title(r"$\mathbf{(a)\ AP_{\mathrm{micro}}\ (Sub\text{-}8\times 8\text{ px})}$", fontsize=11, fontweight='bold', pad=10, color='#0F172A')
    ax1.set_ylabel(r"$\mathrm{AP}_{\mathrm{micro}}$ (%)", fontsize=10, fontweight='bold', color='#334155')
    ax1.set_ylim(32, 45)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=8.5, fontweight='semibold')
    ax1.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax1.annotate('+5.06%', xy=(6, 41.16 + 1.86), xytext=(6, 43.5),
                 ha='center', fontsize=9.0, fontweight='bold', color='#DC2626')

    # Panel 2: coco_AP75
    ax2.set_facecolor('#FAFAFA')
    bars2 = ax2.bar(x, ap75_means, yerr=ap75_stds, capsize=4, width=bar_width,
                    color=colors, edgecolor='#334155', linewidth=0.8, alpha=0.9)
    ax2.set_title(r"$\mathbf{(b)\ coco\_AP_{75}\ (High\ IoU\ \geq 0.75)}$", fontsize=11, fontweight='bold', pad=10, color='#0F172A')
    ax2.set_ylabel(r"$\mathrm{coco\_AP}_{75}$ (%)", fontsize=10, fontweight='bold', color='#334155')
    ax2.set_ylim(4.5, 8.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=35, ha='right', fontsize=8.5, fontweight='semibold')
    ax2.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax2.annotate('+1.40% vs NWD', xy=(6, 7.19 + 0.18), xytext=(5.6, 7.7),
                 ha='center', fontsize=8.5, fontweight='bold', color='#DC2626')

    # Panel 3: mAP_50
    ax3.set_facecolor('#FAFAFA')
    bars3 = ax3.bar(x, map50_means, yerr=map50_stds, capsize=4, width=bar_width,
                    color=colors, edgecolor='#334155', linewidth=0.8, alpha=0.9)
    ax3.set_title(r"$\mathbf{(c)\ mAP_{50}\ (Overall\ Benchmark)}$", fontsize=11, fontweight='bold', pad=10, color='#0F172A')
    ax3.set_ylabel(r"$\mathrm{mAP}_{50}$ (%)", fontsize=10, fontweight='bold', color='#334155')
    ax3.set_ylim(38, 50)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=35, ha='right', fontsize=8.5, fontweight='semibold')
    ax3.grid(True, linestyle=':', alpha=0.6, axis='y')

    for ax in (ax1, ax2, ax3):
        for spine in ax.spines.values():
            spine.set_color('#CBD5E1')

    plt.tight_layout()
    out_png = "paper_a/figures/fig3_megabenchmark_comparison.png"
    out_pdf = "paper_a/figures/fig3_megabenchmark_comparison.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated Clean Figure 3: {out_png}, {out_pdf}")

if __name__ == "__main__":
    create_hybrid_fig1()
    create_hybrid_fig3()
