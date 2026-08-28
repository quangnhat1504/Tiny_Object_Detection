"""
CVPR/ICCV Top-Tier Conference Architecture Diagram Generator for Paper A.
Creates publication-grade, professional vector diagrams matching top-tier 
CVPR/ICCV 2024-2025 visual aesthetics:
- Soft rounded cards, pastel fills, subtle borders, clean hierarchies
- Professional typography, clear mathematical formulations
- Dual output: Native Draw.io (.drawio) XML + Vector PDF + 300 DPI PNG
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
import matplotlib.patheffects as path_effects
import numpy as np

def create_cvpr_figure_1():
    """Generates CVPR-style Fig 1: Joint Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(15.8, 7.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAFA')
    
    # Custom rounded box helper
    def add_card(x, y, w, h, title, subtitle=None, bg_color='#FFFFFF', border_color='#CBD5E1', 
                 title_color='#0F172A', lw=1.5, radius=0.15, shadow=True):
        if shadow:
            # Soft subtle drop shadow
            s_box = FancyBboxPatch((x+0.06, y-0.08), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                   facecolor='#000000', alpha=0.06, edgecolor='none', zorder=1)
            ax.add_patch(s_box)
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
        ax.add_patch(box)
        
        if title:
            ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center',
                    fontsize=10.5, fontweight='bold', color=title_color, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + 0.35, subtitle, ha='center', va='center',
                    fontsize=8.5, fontweight='medium', color='#64748B', zorder=3)
        return box

    # --- Container Regions (Subtle background groupings) ---
    # 1. Backbone Group
    add_card(2.1, 0.4, 3.4, 6.8, "ResNet-50 Backbone", "Bottom-Up Feature Extraction",
             bg_color='#F1F5F9', border_color='#CBD5E1', title_color='#1E293B', lw=1.5, radius=0.25)
    
    # 2. FPN Group
    add_card(6.0, 0.4, 3.2, 6.8, "Feature Pyramid Network (FPN)", "Multi-Scale Top-Down Aggregation",
             bg_color='#F0F9FF', border_color='#BAE6FD', title_color='#0369A1', lw=1.5, radius=0.25)

    # --- 1. Input Image Box ---
    add_card(0.2, 2.4, 1.5, 2.8, "Input Tile", "800 x 800 (b1)",
             bg_color='#FFFFFF', border_color='#94A3B8', title_color='#0F172A', lw=1.5)
    # Mini icon on input
    rect_inner = FancyBboxPatch((0.45, 2.8), 1.0, 1.2, boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor='#E2E8F0', edgecolor='#94A3B8', lw=1, zorder=3)
    ax.add_patch(rect_inner)
    # Tiny target box in image
    ax.add_patch(patches.Rectangle((0.7, 3.2), 0.25, 0.35, facecolor='#FCA5A5', edgecolor='#DC2626', lw=1.2, zorder=4))
    ax.text(0.825, 3.05, "<8x8 px", ha='center', va='center', fontsize=6.5, fontweight='bold', color='#DC2626', zorder=5)

    # --- 2. Backbone Stages (C2 -> C5) ---
    c_stages = [
        ("C2 Stage", "stride 4 | 256-d", 2.3, 1.0, 3.0, 1.1, '#3B82F6', '#EFF6FF', '#BFDBFE'),
        ("C3 Stage", "stride 8 | 512-d", 2.3, 2.4, 3.0, 1.1, '#3B82F6', '#EFF6FF', '#BFDBFE'),
        ("C4 Stage", "stride 16 | 1024-d", 2.3, 3.8, 3.0, 1.1, '#3B82F6', '#EFF6FF', '#BFDBFE'),
        ("C5 Stage", "stride 32 | 2048-d", 2.3, 5.2, 3.0, 1.1, '#3B82F6', '#EFF6FF', '#BFDBFE'),
    ]
    for name, sub, bx, by, bw, bh, tc, bg, bc in c_stages:
        add_card(bx, by, bw, bh, name, sub, bg_color=bg, border_color=bc, title_color=tc)

    # Bottom-up arrows
    for i in range(3):
        ax.annotate("", xy=(3.8, 2.38 + i*1.4), xytext=(3.8, 2.12 + i*1.4),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.35", color="#2563EB", lw=2.0, zorder=4))

    # --- 3. FPN Stages (P2 -> P5) ---
    p_stages = [
        ("P2 Level", "stride 4 | 256-d", 6.2, 1.0, 2.8, 1.1, '#0284C7', '#E0F2FE', '#7DD3FC'),
        ("P3 Level", "stride 8 | 256-d", 6.2, 2.4, 2.8, 1.1, '#0284C7', '#E0F2FE', '#7DD3FC'),
        ("P4 Level", "stride 16 | 256-d", 6.2, 3.8, 2.8, 1.1, '#0284C7', '#E0F2FE', '#7DD3FC'),
        ("P5 Level", "stride 32 | 256-d", 6.2, 5.2, 2.8, 1.1, '#0284C7', '#E0F2FE', '#7DD3FC'),
    ]
    for name, sub, bx, by, bw, bh, tc, bg, bc in p_stages:
        add_card(bx, by, bw, bh, name, sub, bg_color=bg, border_color=bc, title_color=tc)

    # Lateral connections (C_i -> P_i)
    for i in range(4):
        ax.annotate("", xy=(6.18, 1.55 + i*1.4), xytext=(5.32, 1.55 + i*1.4),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.35", color="#0284C7", lw=2.0, zorder=4))
        ax.text(5.75, 1.72 + i*1.4, "1x1 conv", ha='center', va='bottom', fontsize=7, color='#0369A1', zorder=5)

    # Top-down FPN arrows (P5 -> P4 -> P3 -> P2)
    for i in range(3, 0, -1):
        ax.annotate("", xy=(7.6, 1.02 + (i-1)*1.4 + 1.1), xytext=(7.6, 1.02 + i*1.4 - 0.05),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.35", color="#0369A1", lw=2.2, zorder=4))
        ax.text(7.9, 2.22 + (i-1)*1.4, "+ 2x up", ha='left', va='center', fontsize=7, fontweight='bold', color='#0284C7', zorder=5)

    # Input to C2 Arrow
    ax.annotate("", xy=(2.28, 3.8), xytext=(1.72, 3.8),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#475569", lw=2.2, zorder=4))

    # --- 4. Upper Branch: PC-MOC Distillation Engine ---
    add_card(9.7, 5.0, 5.8, 2.2, "PC-MOC Multi-Scale Feature Distillation",
             bg_color='#ECFDF5', border_color='#A7F3D0', title_color='#065F46', lw=1.8, radius=0.2)
    ax.text(12.6, 5.85, r"$\mathcal{L}_{\mathrm{distill}} = \sum_{\ell=2}^{5} \frac{1}{H_\ell W_\ell} \sum_{i,j} \left( 1 - \frac{\langle f_{\mathrm{curr}}^{P_\ell}(i,j),\, f_{\mathrm{ref}}^{P_\ell}(i,j) \rangle}{\|f_{\mathrm{curr}}^{P_\ell}(i,j)\| \|f_{\mathrm{ref}}^{P_\ell}(i,j)\| + \epsilon} \right)$",
            ha='center', va='center', fontsize=8.8, fontweight='bold', color='#047857', zorder=5)
    ax.text(12.6, 5.25, "Aligns FPN representations against reference teacher to prevent curriculum drift",
            ha='center', va='center', fontsize=8, color='#059669', style='italic', zorder=5)

    # Arrow from FPN to PC-MOC
    ax.annotate("", xy=(9.68, 6.1), xytext=(9.02, 5.75),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#059669", lw=2.2, zorder=4))

    # --- 5. Lower Branch: PC-MR Orthogonal Gradient Projection ---
    add_card(9.7, 0.4, 5.8, 2.1, "PC-MR Proposal Micro-Rescue (RPN)",
             bg_color='#FEF2F2', border_color='#FECACA', title_color='#991B1B', lw=1.8, radius=0.2)
    ax.text(12.6, 1.35, r"$\mathbf{g}_{\mathrm{proj}} = \mathbf{g}_{\mathrm{micro}} - \frac{\mathbf{g}_{\mathrm{micro}} \cdot \mathbf{g}_{\mathrm{main}}}{\|\mathbf{g}_{\mathrm{main}}\|^2 + \epsilon} \mathbf{g}_{\mathrm{main}} \quad \text{such that } \mathbf{g}_{\mathrm{proj}} \perp \mathbf{g}_{\mathrm{main}}$",
            ha='center', va='center', fontsize=9.0, fontweight='bold', color='#B91C1C', zorder=5)
    ax.text(12.6, 0.75, "Projects micro-instance gradients onto orthogonal subspace to prevent annihilation",
            ha='center', va='center', fontsize=8, color='#DC2626', style='italic', zorder=5)

    # Arrow from FPN to PC-MR
    ax.annotate("", xy=(9.68, 1.45), xytext=(9.02, 1.55),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#DC2626", lw=2.2, zorder=4))

    # --- 6. Center Branch: Fast R-CNN Head & SA-ALW ---
    add_card(9.7, 2.7, 5.8, 2.1, "RoI Head with SA-ALW & Iterative-CBL",
             bg_color='#FAF5FF', border_color='#E9D5FF', title_color='#6B21A8', lw=1.8, radius=0.2)
    ax.text(12.6, 3.65, r"$\mathcal{L}_{\mathrm{reg}}^{\mathrm{SA}} = \sqrt{w_{\mathrm{pos}}(s) D_{\mathrm{pos}}(p, t) + D_{\mathrm{shape}}(p, t)} \quad\text{with}\quad K_{\mathrm{SA}} = \exp\left[-\beta(s)\sqrt{D_{\mathrm{SA}}}\right]$",
            ha='center', va='center', fontsize=8.6, fontweight='bold', color='#7E22CE', zorder=5)
    ax.text(12.6, 3.05, r"Scale-Adaptive Anisotropic log-Wasserstein with dynamic temperature $\beta(s)$ and weight $w_{\mathrm{pos}}(s)$",
            ha='center', va='center', fontsize=8, color='#9333EA', style='italic', zorder=5)

    # Arrow from FPN to RoI Head
    ax.annotate("", xy=(9.68, 3.75), xytext=(9.02, 3.75),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#9333EA", lw=2.2, zorder=4))

    # Clean border limits
    ax.set_xlim(-0.1, 15.8)
    ax.set_ylim(0.1, 7.5)
    ax.axis('off')
    
    plt.tight_layout()
    out_png = "paper_a/figures/fig1_framework_architecture.png"
    out_pdf = "paper_a/figures/fig1_framework_architecture.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated CVPR-style Fig 1: {out_png}, {out_pdf}")

def create_cvpr_figure_2():
    """Generates CVPR-style Fig 2: Geometry & Loss Landscape Comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.6), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    
    # Range of pixel translations for 6x6 pixel box
    d = np.linspace(0, 10, 200)
    w_t, h_t = 6.0, 6.0
    
    # 1. Standard IoU Loss
    inter = np.maximum(0.0, w_t - d) * h_t
    union = 2 * (w_t * h_t) - inter
    iou = inter / np.maximum(1e-6, union)
    loss_iou = 1.0 - iou
    grad_iou = np.zeros_like(d)
    mask = d < w_t
    grad_iou[mask] = (2 * w_t * h_t) / ((2 * w_t - d[mask])**2 * h_t)
    grad_iou[~mask] = 0.0  # Zero gradient cliff!

    # 2. NWD Gaussian Loss
    C = 12.0
    loss_nwd = 1.0 - np.exp(-d / C)
    grad_nwd = (1.0 / C) * np.exp(-d / C)

    # 3. Proposed SA-ALW Loss
    Sx = (w_t**2 + w_t**2) / 2.0
    w_pos = 2.5
    loss_sa_alw = np.sqrt(w_pos * (d**2 / Sx))
    grad_sa_alw = np.full_like(d, np.sqrt(w_pos / Sx))

    # --- Plot A: Loss Function Profile ---
    ax1.set_facecolor('#FAFAFA')
    ax1.plot(d, loss_iou, label='Standard IoU Loss (Cliff collapse at $d \geq 6$px)', color='#2563EB', lw=2.6, linestyle='--')
    ax1.plot(d, loss_nwd, label='NWD Loss (Isotropic over-smoothed decay)', color='#059669', lw=2.6, linestyle='-.')
    ax1.plot(d, loss_sa_alw, label='Proposed SA-ALW Loss (Smooth, scale-normalized)', color='#7C3AED', lw=3.0)
    
    # Highlight non-overlapping threshold
    ax1.axvline(x=6.0, color='#DC2626', linestyle=':', lw=1.6, alpha=0.8)
    ax1.text(6.1, 0.45, 'Non-overlapping\nthreshold ($d=6$px)', fontsize=8.2, fontweight='bold', color='#DC2626')

    ax1.set_title(r"$\mathbf{(a)\ Loss\ Function\ Value\ vs.\ Positional\ Offset}$", fontsize=11, fontweight='bold', pad=10, color='#0F172A')
    ax1.set_xlabel("Translation Distance $d$ (pixels) on $6\\times 6$ px Box", fontsize=9.5, fontweight='bold', color='#334155')
    ax1.set_ylabel("Bounding Box Regression Loss $\\mathcal{L}(d)$", fontsize=9.5, fontweight='bold', color='#334155')
    ax1.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=8.5)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-0.05, 1.25)

    # --- Plot B: Gradient Magnitude Profile ---
    ax2.set_facecolor('#FAFAFA')
    ax2.plot(d, grad_iou, label='Standard IoU: Zero gradient at $d \geq 6$px', color='#2563EB', lw=2.6, linestyle='--')
    ax2.plot(d, grad_nwd, label='NWD: Exponential decay with distance', color='#059669', lw=2.6, linestyle='-.')
    ax2.plot(d, grad_sa_alw, label='SA-ALW: Non-vanishing stable gradient', color='#7C3AED', lw=3.0)

    # Annotation of zero-gradient annihilation
    ax2.annotate('Gradient Annihilation\n($\\|\\nabla \\mathcal{L}\\| = 0$)', xy=(7.5, 0.0), xytext=(6.5, 0.12),
                 arrowprops=dict(facecolor='#DC2626', shrink=0.08, width=1.5, headwidth=6, edgecolor='none'),
                 fontsize=8.5, fontweight='bold', color='#DC2626')

    ax2.set_title(r"$\mathbf{(b)\ Gradient\ Magnitude\ } \|\nabla_{d} \mathcal{L}\| \mathbf{\ vs.\ Offset}$", fontsize=11, fontweight='bold', pad=10, color='#0F172A')
    ax2.set_xlabel("Translation Distance $d$ (pixels) on $6\\times 6$ px Box", fontsize=9.5, fontweight='bold', color='#334155')
    ax2.set_ylabel("Gradient Magnitude $\\|\\nabla_d \\mathcal{L}\\|$", fontsize=9.5, fontweight='bold', color='#334155')
    ax2.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=8.5)
    ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.02, 0.35)

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color('#CBD5E1')

    plt.tight_layout()
    out_png = "paper_a/figures/fig2_geometry_comparison.png"
    out_pdf = "paper_a/figures/fig2_geometry_comparison.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated CVPR-style Fig 2: {out_png}, {out_pdf}")

def generate_drawio_file():
    """Generates clean, elegant CVPR-style Draw.io XML."""
    xml_content = """<mxfile host="Electron" modified="2026-08-20T08:30:00.000Z" agent="Antigravity CVPR Diagram Synthesizer" version="22.1.18" type="device">
  <diagram id="cvpr_joint_architecture" name="CVPR Joint Architecture">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1654" pageHeight="1169" math="1" shadow="1">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Input Image Card -->
        <mxCell id="input_card" value="&lt;b style=&quot;font-size: 13px;&quot;&gt;Input Image Tile&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #64748B;&quot;&gt;800 × 800 (b1-tiled)&lt;br&gt;Micro-targets &amp;lt; 8×8 px&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.5;shadow=1;fontColor=#0F172A;" vertex="1" parent="1">
          <mxGeometry x="40" y="320" width="160" height="180" as="geometry" />
        </mxCell>

        <!-- ResNet-50 Container -->
        <mxCell id="backbone_container" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;ResNet-50 Backbone&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #1E3A8A;&quot;&gt;Bottom-Up Feature Extraction (Shared Convolutional Weights)&lt;/span&gt;" style="swimlane;whiteSpace=wrap;html=1;rounded=1;arcSize=8;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=1.5;fontColor=#0F172A;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="250" y="160" width="320" height="500" as="geometry" />
        </mxCell>
        <mxCell id="c2" value="&lt;b&gt;C2 Stage&lt;/b&gt;&lt;br&gt;stride 4 | 256-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.2;fontColor=#1E3A8A;" vertex="1" parent="backbone_container">
          <mxGeometry x="25" y="60" width="270" height="85" as="geometry" />
        </mxCell>
        <mxCell id="c3" value="&lt;b&gt;C3 Stage&lt;/b&gt;&lt;br&gt;stride 8 | 512-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.2;fontColor=#1E3A8A;" vertex="1" parent="backbone_container">
          <mxGeometry x="25" y="170" width="270" height="85" as="geometry" />
        </mxCell>
        <mxCell id="c4" value="&lt;b&gt;C4 Stage&lt;/b&gt;&lt;br&gt;stride 16 | 1024-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.2;fontColor=#1E3A8A;" vertex="1" parent="backbone_container">
          <mxGeometry x="25" y="280" width="270" height="85" as="geometry" />
        </mxCell>
        <mxCell id="c5" value="&lt;b&gt;C5 Stage&lt;/b&gt;&lt;br&gt;stride 32 | 2048-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.2;fontColor=#1E3A8A;" vertex="1" parent="backbone_container">
          <mxGeometry x="25" y="390" width="270" height="85" as="geometry" />
        </mxCell>

        <!-- FPN Container -->
        <mxCell id="fpn_container" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;Feature Pyramid Network (FPN)&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #0369A1;&quot;&gt;Top-Down Multi-Scale Semantic Aggregation&lt;/span&gt;" style="swimlane;whiteSpace=wrap;html=1;rounded=1;arcSize=8;fillColor=#F0F9FF;strokeColor=#BAE6FD;strokeWidth=1.5;fontColor=#0369A1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="630" y="160" width="300" height="500" as="geometry" />
        </mxCell>
        <mxCell id="p2" value="&lt;b&gt;P2 Level&lt;/b&gt;&lt;br&gt;stride 4 | 256-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.2;fontColor=#0369A1;" vertex="1" parent="fpn_container">
          <mxGeometry x="25" y="60" width="250" height="85" as="geometry" />
        </mxCell>
        <mxCell id="p3" value="&lt;b&gt;P3 Level&lt;/b&gt;&lt;br&gt;stride 8 | 256-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.2;fontColor=#0369A1;" vertex="1" parent="fpn_container">
          <mxGeometry x="25" y="170" width="250" height="85" as="geometry" />
        </mxCell>
        <mxCell id="p4" value="&lt;b&gt;P4 Level&lt;/b&gt;&lt;br&gt;stride 16 | 256-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.2;fontColor=#0369A1;" vertex="1" parent="fpn_container">
          <mxGeometry x="25" y="280" width="250" height="85" as="geometry" />
        </mxCell>
        <mxCell id="p5" value="&lt;b&gt;P5 Level&lt;/b&gt;&lt;br&gt;stride 32 | 256-d" style="rounded=1;whiteSpace=wrap;html=1;arcSize=10;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.2;fontColor=#0369A1;" vertex="1" parent="fpn_container">
          <mxGeometry x="25" y="390" width="250" height="85" as="geometry" />
        </mxCell>

        <!-- Upper Module: PC-MOC Distillation -->
        <mxCell id="pcmoc_card" value="&lt;b style=&quot;font-size: 13px;&quot;&gt;PC-MOC Multi-Scale Feature Distillation&lt;/b&gt;&lt;br&gt;$$\mathcal{L}_{\mathrm{distill}} = \sum_{\ell=2}^5 \frac{1}{H_\ell W_\ell} \left(1 - \cos(f_{\mathrm{curr}}^{P_\ell}, f_{\mathrm{ref}}^{P_\ell})\right)$$&lt;br&gt;&lt;span style=&quot;color: #059669; font-size: 11px;&quot;&gt;Gradient-stabilized cosine feature alignment to prevent curriculum drift&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#ECFDF5;strokeColor=#A7F3D0;strokeWidth=1.8;fontColor=#065F46;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="990" y="160" width="580" height="120" as="geometry" />
        </mxCell>

        <!-- Center Module: RoI Head & SA-ALW -->
        <mxCell id="roi_card" value="&lt;b style=&quot;font-size: 13px;&quot;&gt;RoI Head with SA-ALW &amp;amp; Iterative-CBL&lt;/b&gt;&lt;br&gt;$$\mathcal{L}_{\mathrm{reg}}^{\mathrm{SA}} = \sqrt{w_{\mathrm{pos}}(s) D_{\mathrm{pos}} + D_{\mathrm{shape}}} \quad\text{with}\quad K_{\mathrm{SA}} = \exp\left[-\beta(s)\sqrt{D_{\mathrm{SA}}}\right]$$&lt;br&gt;&lt;span style=&quot;color: #7E22CE; font-size: 11px;&quot;&gt;Scale-Adaptive Anisotropic log-Wasserstein with dynamic temperature \beta(s)&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#FAF5FF;strokeColor=#E9D5FF;strokeWidth=1.8;fontColor=#6B21A8;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="990" y="340" width="580" height="140" as="geometry" />
        </mxCell>

        <!-- Lower Module: PC-MR Gradient Projection -->
        <mxCell id="pcmr_card" value="&lt;b style=&quot;font-size: 13px;&quot;&gt;PC-MR Proposal Micro-Rescue (RPN)&lt;/b&gt;&lt;br&gt;$$\mathbf{g}_{\mathrm{proj}} = \mathbf{g}_{\mathrm{micro}} - \frac{\mathbf{g}_{\mathrm{micro}} \cdot \mathbf{g}_{\mathrm{main}}}{\|\mathbf{g}_{\mathrm{main}}\|^2 + \epsilon} \mathbf{g}_{\mathrm{main}} \quad (\mathbf{g}_{\mathrm{proj}} \perp \mathbf{g}_{\mathrm{main}})$$&lt;br&gt;&lt;span style=&quot;color: #DC2626; font-size: 11px;&quot;&gt;Orthogonal projection prevents micro-proposal gradient cancellation&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;arcSize=8;fillColor=#FEF2F2;strokeColor=#FECACA;strokeWidth=1.8;fontColor=#991B1B;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="990" y="530" width="580" height="130" as="geometry" />
        </mxCell>

        <!-- Connectors -->
        <mxCell id="e_in" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#475569;strokeWidth=2.2;html=1;" edge="1" parent="1" source="input_card" target="backbone_container">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_lat" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0284C7;strokeWidth=2.2;html=1;dashed=1;" edge="1" parent="1" source="backbone_container" target="fpn_container">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_moc" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#059669;strokeWidth=2.5;html=1;" edge="1" parent="1" source="fpn_container" target="pcmoc_card">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_roi" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#9333EA;strokeWidth=2.5;html=1;" edge="1" parent="1" source="fpn_container" target="roi_card">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_mr" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#DC2626;strokeWidth=2.5;html=1;" edge="1" parent="1" source="fpn_container" target="pcmr_card">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    out_drawio = "paper_a/figures/fig1_framework_architecture.drawio"
    with open(out_drawio, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"Generated CVPR-style Draw.io XML: {out_drawio}")

if __name__ == "__main__":
    create_cvpr_figure_1()
    create_cvpr_figure_2()
    generate_drawio_file()
