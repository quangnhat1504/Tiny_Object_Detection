"""
Masterpiece CVPR/ICCV Architecture Diagram Generator for Paper A.
Engineered for:
- Maximum Visual Balance & Harmony across 4 structured columns
- High-readability large mathematical typography (15pt bold LaTeX font)
- Comprehensive pipeline overview: Input Tile -> ResNet-50 -> FPN -> PC-MOC / SA-ALW / PC-MR
- Embedded photorealistic aerial tile with micro-target zoom callout
- Native Draw.io XML export (.drawio)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
from PIL import Image
import numpy as np

def create_masterpiece_architecture():
    # Large 19.5 x 8.8 inch canvas for crisp 300 DPI raster and vector PDF
    fig, ax = plt.subplots(figsize=(19.5, 8.8), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#F8FAFC')

    def add_card(x, y, w, h, title=None, subtitle=None, bg_color='#FFFFFF', border_color='#CBD5E1', 
                 title_color='#0F172A', lw=1.6, radius=0.22, shadow=True, title_fontsize=13.0, sub_fontsize=10.0):
        if shadow:
            s_box = FancyBboxPatch((x+0.06, y-0.07), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                   facecolor='#000000', alpha=0.06, edgecolor='none', zorder=1)
            ax.add_patch(s_box)
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                             facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2)
        ax.add_patch(box)
        
        if title:
            ax.text(x + w/2, y + h - 0.38, title, ha='center', va='center',
                    fontsize=title_fontsize, fontweight='bold', color=title_color, zorder=5)
        if subtitle:
            ax.text(x + w/2, y + h - 0.75, subtitle, ha='center', va='center',
                    fontsize=sub_fontsize, fontweight='medium', color='#64748B', zorder=5)
        return box

    # =========================================================================
    # COLUMN 1: INPUT IMAGE TILE & MICROSCOPIC ZOOM INSET (x in [0.4, 4.0])
    # =========================================================================
    col1_w = 3.6
    add_card(0.4, 0.4, col1_w, 8.0, "Input Image Tile (b1-tiled)", "TinyPerson 800 x 800 px",
             bg_color='#FFFFFF', border_color='#94A3B8', title_color='#0F172A', lw=1.8,
             title_fontsize=13.0, sub_fontsize=10.0)

    # Embedded AI Aerial Tile
    aerial_path = "paper_a/figures/assets/aerial_tile.jpg"
    if os.path.exists(aerial_path):
        img_aerial = Image.open(aerial_path)
        ax.imshow(img_aerial, extent=[0.6, 3.8, 4.2, 7.4], zorder=3, aspect='auto')
        # Micro bounding box on image
        ax.add_patch(Rectangle((2.2, 5.8), 0.25, 0.25, edgecolor='#DC2626', facecolor='none', lw=2.2, zorder=4))
        # Normal bounding box on image
        ax.add_patch(Rectangle((3.1, 6.5), 0.45, 0.30, edgecolor='#2563EB', facecolor='none', lw=1.8, zorder=4))
        ax.text(3.32, 6.95, "Normal Target", ha='center', fontsize=8.0, fontweight='bold', color='#1D4ED8', zorder=5)

    # Zoom-In Inset Container
    add_card(0.6, 0.7, 3.2, 3.2, "Microscopic Target Callout", "Spatial Discretization Noise Zone",
             bg_color='#FEF2F2', border_color='#FCA5A5', title_color='#DC2626', lw=1.6, radius=0.16,
             title_fontsize=11.5, sub_fontsize=9.0)

    # Pixel Grid simulation
    grid_x, grid_y, grid_s = 0.8, 1.0, 1.4
    for gx in np.linspace(grid_x, grid_x + grid_s, 6):
        ax.plot([gx, gx], [grid_y, grid_y + grid_s], color='#FCA5A5', lw=0.7, zorder=3)
    for gy in np.linspace(grid_y, grid_y + grid_s, 6):
        ax.plot([grid_x, grid_x + grid_s], [gy, gy], color='#FCA5A5', lw=0.7, zorder=3)
    
    # Target Box on grid
    ax.add_patch(Rectangle((grid_x + 0.28, grid_y + 0.28), 0.84, 0.84,
                           edgecolor='#DC2626', facecolor='#FEE2E2', lw=2.2, alpha=0.85, zorder=4))
    ax.text(grid_x + 0.70, grid_y + 0.70, "Target\n6x6 px", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#991B1B', zorder=5)

    # Callout Annotation Text
    ax.text(2.2, 3.0, r"$\mathbf{Target\ Scale:\ } s \leq 8\text{ px}$" + "\n" +
                      r"$\bullet\ \text{Severe IoU Cliff Collapse}$" + "\n" +
                      r"$\bullet\ \nabla_{\boldsymbol{\theta}} \mathcal{L}_{\mathrm{IoU}} = \mathbf{0}\text{ at } d \geq 6\text{px}$",
            ha='left', va='top', fontsize=9.2, fontweight='bold', color='#7F1D1D', zorder=5)

    # =========================================================================
    # COLUMN 2: RESNET-50 BACKBONE (x in [4.4, 7.8])
    # =========================================================================
    col2_w = 3.4
    add_card(4.4, 0.4, col2_w, 8.0, "ResNet-50 Backbone", "Bottom-Up Shared Feature Extractor",
             bg_color='#F1F5F9', border_color='#CBD5E1', title_color='#1E293B', lw=1.8,
             title_fontsize=13.0, sub_fontsize=10.0)

    # 4 Stacked Stages
    c_stages = [
        ("C2 Stage", "stride 4  |  H/4 x W/4 x 256", 4.6, 1.0, 3.0, 1.3),
        ("C3 Stage", "stride 8  |  H/8 x W/8 x 512", 4.6, 2.6, 3.0, 1.3),
        ("C4 Stage", "stride 16 |  H/16 x W/16 x 1024", 4.6, 4.2, 3.0, 1.3),
        ("C5 Stage", "stride 32 |  H/32 x W/32 x 2048", 4.6, 5.8, 3.0, 1.3),
    ]
    for name, sub, bx, by, bw, bh in c_stages:
        add_card(bx, by, bw, bh, name, sub, bg_color='#EFF6FF', border_color='#93C5FD',
                 title_color='#1E3A8A', lw=1.4, title_fontsize=11.5, sub_fontsize=9.0)

    # Upward backbone flow arrows
    for i in range(3):
        ax.annotate("", xy=(6.1, 2.58 + i*1.6), xytext=(6.1, 2.32 + i*1.6),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#2563EB", lw=2.4, zorder=6))

    # Input to C2 Arrow
    ax.annotate("", xy=(4.38, 4.4), xytext=(4.02, 4.4),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45", color="#334155", lw=2.6, zorder=6))

    # =========================================================================
    # COLUMN 3: FEATURE PYRAMID NETWORK (FPN) (x in [8.2, 11.6])
    # =========================================================================
    col3_w = 3.4
    add_card(8.2, 0.4, col3_w, 8.0, "Feature Pyramid Network", "Top-Down Multi-Scale Semantic Pyramids",
             bg_color='#F0F9FF', border_color='#BAE6FD', title_color='#0369A1', lw=1.8,
             title_fontsize=13.0, sub_fontsize=10.0)

    p_stages = [
        ("P2 Level", "stride 4  |  256-d (Highest Res)", 8.4, 1.0, 3.0, 1.3),
        ("P3 Level", "stride 8  |  256-d (Small Objects)", 8.4, 2.6, 3.0, 1.3),
        ("P4 Level", "stride 16 |  256-d (Medium Objects)", 8.4, 4.2, 3.0, 1.3),
        ("P5 Level", "stride 32 |  256-d (Large Objects)", 8.4, 5.8, 3.0, 1.3),
    ]
    for name, sub, bx, by, bw, bh in p_stages:
        add_card(bx, by, bw, bh, name, sub, bg_color='#E0F2FE', border_color='#7DD3FC',
                 title_color='#0284C7', lw=1.4, title_fontsize=11.5, sub_fontsize=9.0)

    # Lateral 1x1 Conv Arrows (C_i -> P_i)
    for i in range(4):
        ax.annotate("", xy=(8.38, 1.65 + i*1.6), xytext=(7.62, 1.65 + i*1.6),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.35", color="#0284C7", lw=2.0, zorder=6))
        ax.text(8.0, 1.85 + i*1.6, "1x1 conv", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#0369A1', zorder=7)

    # Top-Down FPN Arrows (P5 -> P4 -> P3 -> P2)
    for i in range(3, 0, -1):
        ax.annotate("", xy=(9.9, 1.02 + (i-1)*1.6 + 1.3), xytext=(9.9, 1.02 + i*1.6 - 0.05),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#0369A1", lw=2.4, zorder=6))
        ax.text(10.25, 2.45 + (i-1)*1.6, "+ 2x up", ha='left', va='center', fontsize=8.0, fontweight='bold', color='#0284C7', zorder=7)

    # =========================================================================
    # COLUMN 4: THREE PROPOSED CORE INNOVATIONS (x in [12.0, 19.1])
    # =========================================================================
    col4_w = 7.1
    
    # -------------------------------------------------------------------------
    # Branch 1: Upper Card - PC-MOC Multi-Scale Feature Distillation
    # -------------------------------------------------------------------------
    add_card(12.0, 5.8, col4_w, 2.6, 
             "PC-MOC Multi-Scale Cosine Feature Distillation",
             "Prevents multi-scale representation drift during curriculum scale updates",
             bg_color='#ECFDF5', border_color='#A7F3D0', title_color='#065F46', lw=2.0,
             title_fontsize=13.0, sub_fontsize=10.0)

    # Large prominent math formula
    ax.text(15.55, 6.75, 
            r"$\mathcal{L}_{\mathrm{distill}} = \sum_{\ell=2}^{5} \frac{1}{H_\ell W_\ell} \sum_{i,j} \left( 1 - \frac{\langle f_{\mathrm{curr}}^{P_\ell}(i,j),\, f_{\mathrm{ref}}^{P_\ell}(i,j) \rangle}{\|f_{\mathrm{curr}}^{P_\ell}(i,j)\| \|f_{\mathrm{ref}}^{P_\ell}(i,j)\| + \epsilon} \right)$",
            ha='center', va='center', fontsize=14.0, fontweight='bold', color='#047857', zorder=6)

    ax.text(15.55, 6.15, 
            r"$\bullet\ f_{\mathrm{curr}}^{P_\ell}\text{: Active student FPN features} \qquad \bullet\ f_{\mathrm{ref}}^{P_\ell}\text{: Teacher reference representation}$",
            ha='center', va='center', fontsize=10.0, fontweight='bold', color='#065F46', zorder=6)

    # Arrow from FPN to PC-MOC
    ax.annotate("", xy=(11.98, 7.1), xytext=(11.42, 6.45),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45", color="#059669", lw=2.6, zorder=6))

    # -------------------------------------------------------------------------
    # Branch 2: Center Card - RoI Head with SA-ALW & Iterative-CBL
    # -------------------------------------------------------------------------
    add_card(12.0, 3.0, col4_w, 2.6, 
             "RoI Head with SA-ALW & Iterative Curriculum Routing",
             "Scale-Adaptive Anisotropic Log-Wasserstein distance with scale-conditioned schedules",
             bg_color='#FAF5FF', border_color='#E9D5FF', title_color='#6B21A8', lw=2.0,
             title_fontsize=13.0, sub_fontsize=10.0)

    # Large prominent math formulas
    ax.text(15.55, 4.05, 
            r"$\mathcal{L}_{\mathrm{reg}}^{\mathrm{SA}} = \sqrt{w_{\mathrm{pos}}(s) D_{\mathrm{pos}}(p, t) + D_{\mathrm{shape}}(p, t)} \qquad K_{\mathrm{SA}} = \exp\left[ -\beta(s) \sqrt{D_{\mathrm{SA}}(p, t)} \right]$",
            ha='center', va='center', fontsize=13.5, fontweight='bold', color='#7E22CE', zorder=6)

    ax.text(15.55, 3.40, 
            r"$\bullet\ \text{Adaptive temperature: } \beta(s) = \beta_{\min} + \Delta\beta \cdot u(s) \qquad \bullet\ \text{Position weight: } w_{\mathrm{pos}}(s) = w_{\min} + \Delta w \cdot u(s)$",
            ha='center', va='center', fontsize=10.0, fontweight='bold', color='#6B21A8', zorder=6)

    # Arrow from FPN to RoI Head
    ax.annotate("", xy=(11.98, 4.3), xytext=(11.42, 4.3),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45", color="#9333EA", lw=2.6, zorder=6))

    # -------------------------------------------------------------------------
    # Branch 3: Lower Card - PC-MR Proposal Micro-Rescue (RPN)
    # -------------------------------------------------------------------------
    add_card(12.0, 0.4, col4_w, 2.4, 
             "PC-MR Proposal Micro-Rescue with Orthogonal Gradient Projection",
             "Eliminates gradient cancellation between dominant anchors and microscopic proposals",
             bg_color='#FEF2F2', border_color='#FECACA', title_color='#991B1B', lw=2.0,
             title_fontsize=13.0, sub_fontsize=10.0)

    # Large prominent math formula
    ax.text(15.55, 1.45, 
            r"$\mathbf{g}_{\mathrm{proj}} = \mathbf{g}_{\mathrm{micro}} - \frac{\mathbf{g}_{\mathrm{micro}} \cdot \mathbf{g}_{\mathrm{main}}}{\|\mathbf{g}_{\mathrm{main}}\|^2 + \epsilon} \mathbf{g}_{\mathrm{main}} \qquad (\mathbf{g}_{\mathrm{proj}} \perp \mathbf{g}_{\mathrm{main}})$",
            ha='center', va='center', fontsize=14.0, fontweight='bold', color='#B91C1C', zorder=6)

    ax.text(15.55, 0.85, 
            r"$\bullet\ \mathbf{g}_{\mathrm{total}} = \mathbf{g}_{\mathrm{main}} + \lambda_{\mathrm{MR}} \mathbf{g}_{\mathrm{proj}} \quad \text{guarantees zero gradient annihilation on microscopic targets}$",
            ha='center', va='center', fontsize=10.0, fontweight='bold', color='#991B1B', zorder=6)

    # Arrow from FPN to PC-MR
    ax.annotate("", xy=(11.98, 1.6), xytext=(11.42, 2.15),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45", color="#DC2626", lw=2.6, zorder=6))

    # Canvas boundaries
    ax.set_xlim(0.1, 19.3)
    ax.set_ylim(0.1, 8.6)
    ax.axis('off')

    plt.tight_layout()
    out_png = "paper_a/figures/fig1_framework_architecture.png"
    out_pdf = "paper_a/figures/fig1_framework_architecture.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated Masterpiece Architecture Fig 1: {out_png}, {out_pdf}")

def generate_masterpiece_drawio():
    xml = """<mxfile host="Electron" modified="2026-08-20T08:45:00.000Z" agent="Antigravity Masterpiece Synthesizer" version="22.1.18" type="device">
  <diagram id="masterpiece_architecture" name="Masterpiece CVPR Joint Architecture">
    <mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1920" pageHeight="1080" math="1" shadow="1">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- Col 1: Input Stage -->
        <mxCell id="col1" value="&lt;b style=&quot;font-size: 15px;&quot;&gt;Input Image Tile (b1-tiled)&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #64748B;&quot;&gt;TinyPerson 800 × 800 px&lt;/span&gt;" style="swimlane;whiteSpace=wrap;html=1;rounded=1;arcSize=8;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=2;fontColor=#0F172A;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="40" y="80" width="340" height="760" as="geometry" />
        </mxCell>
        <mxCell id="aerial_img" value="[Aerial Drone Maritime Tile Image]" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E2E8F0;strokeColor=#94A3B8;fontColor=#334155;fontStyle=1;" vertex="1" parent="col1">
          <mxGeometry x="25" y="60" width="290" height="300" as="geometry" />
        </mxCell>
        <mxCell id="zoom_box" value="&lt;b style=&quot;font-size: 13px; color: #DC2626;&quot;&gt;Microscopic Target Callout&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #7F1D1D; font-size: 11px;&quot;&gt;Spatial Discretization Noise Zone&lt;br&gt;&lt;b&gt;Target Scale: s ≤ 8 px&lt;/b&gt;&lt;br&gt;Severe IoU Step Collapse&lt;br&gt;∇IoU = 0 at d ≥ 6 px&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#FCA5A5;strokeWidth=1.8;arcSize=10;" vertex="1" parent="col1">
          <mxGeometry x="25" y="380" width="290" height="340" as="geometry" />
        </mxCell>

        <!-- Col 2: ResNet-50 Backbone -->
        <mxCell id="col2" value="&lt;b style=&quot;font-size: 15px;&quot;&gt;ResNet-50 Backbone&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #1E3A8A;&quot;&gt;Bottom-Up Shared Feature Extractor&lt;/span&gt;" style="swimlane;whiteSpace=wrap;html=1;rounded=1;arcSize=8;fillColor=#F1F5F9;strokeColor=#CBD5E1;strokeWidth=2;fontColor=#1E293B;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="420" y="80" width="320" height="760" as="geometry" />
        </mxCell>
        <mxCell id="c2" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;C2 Stage&lt;/b&gt;&lt;br&gt;stride 4  |  H/4 × W/4 × 256" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.5;fontColor=#1E3A8A;arcSize=10;" vertex="1" parent="col2">
          <mxGeometry x="25" y="600" width="270" height="110" as="geometry" />
        </mxCell>
        <mxCell id="c3" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;C3 Stage&lt;/b&gt;&lt;br&gt;stride 8  |  H/8 × W/8 × 512" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.5;fontColor=#1E3A8A;arcSize=10;" vertex="1" parent="col2">
          <mxGeometry x="25" y="440" width="270" height="110" as="geometry" />
        </mxCell>
        <mxCell id="c4" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;C4 Stage&lt;/b&gt;&lt;br&gt;stride 16 |  H/16 × W/16 × 1024" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.5;fontColor=#1E3A8A;arcSize=10;" vertex="1" parent="col2">
          <mxGeometry x="25" y="280" width="270" height="110" as="geometry" />
        </mxCell>
        <mxCell id="c5" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;C5 Stage&lt;/b&gt;&lt;br&gt;stride 32 |  H/32 × W/32 × 2048" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#93C5FD;strokeWidth=1.5;fontColor=#1E3A8A;arcSize=10;" vertex="1" parent="col2">
          <mxGeometry x="25" y="120" width="270" height="110" as="geometry" />
        </mxCell>

        <!-- Col 3: FPN Pyramids -->
        <mxCell id="col3" value="&lt;b style=&quot;font-size: 15px;&quot;&gt;Feature Pyramid Network&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #0369A1;&quot;&gt;Top-Down Multi-Scale Semantic Pyramids&lt;/span&gt;" style="swimlane;whiteSpace=wrap;html=1;rounded=1;arcSize=8;fillColor=#F0F9FF;strokeColor=#BAE6FD;strokeWidth=2;fontColor=#0369A1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="780" y="80" width="320" height="760" as="geometry" />
        </mxCell>
        <mxCell id="p2" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;P2 Level&lt;/b&gt;&lt;br&gt;stride 4  |  256-d (Highest Res)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.5;fontColor=#0284C7;arcSize=10;" vertex="1" parent="col3">
          <mxGeometry x="25" y="600" width="270" height="110" as="geometry" />
        </mxCell>
        <mxCell id="p3" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;P3 Level&lt;/b&gt;&lt;br&gt;stride 8  |  256-d (Small Objects)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.5;fontColor=#0284C7;arcSize=10;" vertex="1" parent="col3">
          <mxGeometry x="25" y="440" width="270" height="110" as="geometry" />
        </mxCell>
        <mxCell id="p4" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;P4 Level&lt;/b&gt;&lt;br&gt;stride 16 |  256-d (Medium Objects)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.5;fontColor=#0284C7;arcSize=10;" vertex="1" parent="col3">
          <mxGeometry x="25" y="280" width="270" height="110" as="geometry" />
        </mxCell>
        <mxCell id="p5" value="&lt;b style=&quot;font-size: 14px;&quot;&gt;P5 Level&lt;/b&gt;&lt;br&gt;stride 32 |  256-d (Large Objects)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E0F2FE;strokeColor=#7DD3FC;strokeWidth=1.5;fontColor=#0284C7;arcSize=10;" vertex="1" parent="col3">
          <mxGeometry x="25" y="120" width="270" height="110" as="geometry" />
        </mxCell>

        <!-- Col 4: Three Core Innovations -->
        <mxCell id="pcmoc_card" value="&lt;b style=&quot;font-size: 15px; color: #065F46;&quot;&gt;PC-MOC Multi-Scale Cosine Feature Distillation&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #059669; font-size: 12px;&quot;&gt;Prevents multi-scale representation drift during curriculum scale updates&lt;/span&gt;&lt;br&gt;&lt;br&gt;&lt;span style=&quot;font-size: 16px; font-weight: bold; color: #047857;&quot;&gt;$$\mathcal{L}_{\mathrm{distill}} = \sum_{\ell=2}^{5} \frac{1}{H_\ell W_\ell} \sum_{i,j} \left( 1 - \frac{\langle f_{\mathrm{curr}}^{P_\ell}(i,j),\, f_{\mathrm{ref}}^{P_\ell}(i,j) \rangle}{\|f_{\mathrm{curr}}^{P_\ell}(i,j)\| \|f_{\mathrm{ref}}^{P_\ell}(i,j)\| + \epsilon} \right)$$&lt;/span&gt;&lt;br&gt;&lt;span style=&quot;color: #065F46; font-size: 11px;&quot;&gt;• f_curr: Student FPN Features  |  • f_ref: Teacher Reference Model&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ECFDF5;strokeColor=#A7F3D0;strokeWidth=2.2;fontColor=#065F46;shadow=1;arcSize=8;" vertex="1" parent="1">
          <mxGeometry x="1140" y="80" width="740" height="230" as="geometry" />
        </mxCell>

        <mxCell id="roi_card" value="&lt;b style=&quot;font-size: 15px; color: #6B21A8;&quot;&gt;RoI Head with SA-ALW &amp;amp; Iterative Curriculum Routing&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #7E22CE; font-size: 12px;&quot;&gt;Scale-Adaptive Anisotropic Log-Wasserstein distance with scale-conditioned schedules&lt;/span&gt;&lt;br&gt;&lt;br&gt;&lt;span style=&quot;font-size: 15px; font-weight: bold; color: #7E22CE;&quot;&gt;$$\mathcal{L}_{\mathrm{reg}}^{\mathrm{SA}} = \sqrt{w_{\mathrm{pos}}(s) D_{\mathrm{pos}}(p, t) + D_{\mathrm{shape}}(p, t)} \qquad K_{\mathrm{SA}} = \exp\left[ -\beta(s)\sqrt{D_{\mathrm{SA}}(p, t)} \right]$$&lt;/span&gt;&lt;br&gt;&lt;span style=&quot;color: #6B21A8; font-size: 11px;&quot;&gt;• β(s) = β_min + Δβ · u(s)  |  • w_pos(s) = w_min + Δw · u(s)&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FAF5FF;strokeColor=#E9D5FF;strokeWidth=2.2;fontColor=#6B21A8;shadow=1;arcSize=8;" vertex="1" parent="1">
          <mxGeometry x="1140" y="340" width="740" height="240" as="geometry" />
        </mxCell>

        <mxCell id="pcmr_card" value="&lt;b style=&quot;font-size: 15px; color: #991B1B;&quot;&gt;PC-MR Proposal Micro-Rescue with Orthogonal Gradient Projection&lt;/b&gt;&lt;br&gt;&lt;span style=&quot;color: #B91C1C; font-size: 12px;&quot;&gt;Eliminates gradient cancellation between dominant anchors and microscopic proposals&lt;/span&gt;&lt;br&gt;&lt;br&gt;&lt;span style=&quot;font-size: 16px; font-weight: bold; color: #B91C1C;&quot;&gt;$$\mathbf{g}_{\mathrm{proj}} = \mathbf{g}_{\mathrm{micro}} - \frac{\mathbf{g}_{\mathrm{micro}} \cdot \mathbf{g}_{\mathrm{main}}}{\|\mathbf{g}_{\mathrm{main}}\|^2 + \epsilon} \mathbf{g}_{\mathrm{main}} \quad (\mathbf{g}_{\mathrm{proj}} \perp \mathbf{g}_{\mathrm{main}})$$&lt;/span&gt;&lt;br&gt;&lt;span style=&quot;color: #991B1B; font-size: 11px;&quot;&gt;• g_total = g_main + λ_MR · g_proj guarantees zero gradient annihilation on microscopic targets&lt;/span&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#FECACA;strokeWidth=2.2;fontColor=#991B1B;shadow=1;arcSize=8;" vertex="1" parent="1">
          <mxGeometry x="1140" y="610" width="740" height="230" as="geometry" />
        </mxCell>

        <!-- Connectors -->
        <mxCell id="e_in" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#334155;strokeWidth=2.8;html=1;" edge="1" parent="1" source="col1" target="col2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_fpn" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#0284C7;strokeWidth=2.8;dashed=1;html=1;" edge="1" parent="1" source="col2" target="col3">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_moc" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#059669;strokeWidth=3;html=1;" edge="1" parent="1" source="col3" target="pcmoc_card">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_roi" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#9333EA;strokeWidth=3;html=1;" edge="1" parent="1" source="col3" target="roi_card">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="e_mr" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#DC2626;strokeWidth=3;html=1;" edge="1" parent="1" source="col3" target="pcmr_card">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    out_drawio = "paper_a/figures/fig1_framework_architecture.drawio"
    with open(out_drawio, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"Generated Masterpiece Draw.io XML: {out_drawio}")

if __name__ == "__main__":
    create_masterpiece_architecture()
    generate_masterpiece_drawio()
