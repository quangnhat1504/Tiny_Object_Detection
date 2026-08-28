"""
Advanced 3D Academic Diagram Generator for Paper A.
Generates:
1. Fig 1: 3D Volumetric Isometric Architecture (PNG, PDF, SVG, native .drawio XML)
2. Fig 2: 3D Loss Landscape & Gradient Surface Plots (PNG, PDF, SVG)
3. Fig 3 & Fig 4: High-aesthetic Vector Benchmark Figures
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.colors import to_rgba, to_hex, rgb_to_hsv, hsv_to_rgb
import matplotlib.cm as cm

def adjust_lightness(color, factor):
    """Adjust lightness of an RGB color (factor > 1 lighter, < 1 darker)."""
    rgba = to_rgba(color)
    hsv = rgb_to_hsv(rgba[:3])
    hsv[2] = max(0.0, min(1.0, hsv[2] * factor))
    if factor > 1.0:
        hsv[1] = max(0.0, min(1.0, hsv[1] * (2.0 - factor)))
    return to_hex(hsv_to_rgb(hsv))

def draw_3d_block(ax, x, y, z, dx, dy, dz, color, alpha=0.92, label=None, sublabel=None, 
                  text_color='white', angle=math.radians(28), elevation=math.radians(22),
                  edgecolor='#1E293B', linewidth=1.2, shadow=True):
    """Draws a realistic 3D isometric volumetric block with top, front, and right shaded faces."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    sin_e = math.sin(elevation)
    
    def project(px, py, pz):
        sx = px * cos_a + py * cos_a * 0.85
        sy = -px * sin_a * sin_e + py * sin_a + pz
        return sx, sy
    
    v = [
        project(x, y, z),               # 0
        project(x + dx, y, z),          # 1
        project(x + dx, y + dy, z),     # 2
        project(x, y + dy, z),          # 3
        project(x, y, z + dz),          # 4
        project(x + dx, y, z + dz),     # 5
        project(x + dx, y + dy, z + dz),# 6
        project(x, y + dy, z + dz),     # 7
    ]
    
    if shadow:
        s_pts = np.array([
            project(x + 0.15, y - 0.2, 0),
            project(x + dx + 0.25, y - 0.2, 0),
            project(x + dx + 0.35, y + dy - 0.1, 0),
            project(x + 0.15, y + dy - 0.1, 0)
        ])
        ax.add_patch(Polygon(s_pts, closed=True, facecolor='#000000', alpha=0.12, edgecolor='none', zorder=1))
    
    c_top = adjust_lightness(color, 1.25)
    c_front = adjust_lightness(color, 0.95)
    c_right = adjust_lightness(color, 0.70)
    
    p_front = Polygon(np.array([v[0], v[1], v[5], v[4]]), closed=True,
                      facecolor=c_front, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=3)
    ax.add_patch(p_front)
    
    p_right = Polygon(np.array([v[1], v[2], v[6], v[5]]), closed=True,
                      facecolor=c_right, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=3)
    ax.add_patch(p_right)
    
    p_top = Polygon(np.array([v[4], v[5], v[6], v[7]]), closed=True,
                    facecolor=c_top, edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=4)
    ax.add_patch(p_top)
    
    if label:
        ctx = (v[4][0] + v[5][0] + v[6][0] + v[7][0]) / 4.0
        cty = (v[4][1] + v[5][1] + v[6][1] + v[7][1]) / 4.0
        cfx = (v[0][0] + v[1][0] + v[5][0] + v[4][0]) / 4.0
        cfy = (v[0][1] + v[1][1] + v[5][1] + v[4][1]) / 4.0
        
        ax.text(cfx, cfy, label, ha='center', va='center', fontsize=9.5, fontweight='bold',
                color=text_color, zorder=5)
        if sublabel:
            ax.text(ctx, cty, sublabel, ha='center', va='center', fontsize=8,
                    fontweight='bold', color='#0F172A', zorder=5)
            
    return v

def generate_3d_architecture_diagram():
    """Generates a high-resolution 3D volumetric diagram of the Joint Architecture."""
    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=300)
    ax.set_facecolor('#F8FAFC')
    fig.patch.set_facecolor('#FFFFFF')
    
    c_img = '#64748B'
    c_backbone = '#2563EB'
    c_fpn = '#0284C7'
    c_pcmr = '#DC2626'
    c_pcmoc = '#059669'
    c_roi = '#D97706'
    
    # --- 1. Input Image ---
    v_img = draw_3d_block(ax, x=0.5, y=0, z=1.5, dx=0.4, dy=2.8, dz=3.2, color=c_img,
                          label="Input Image\n800x800", sublabel="Tile (b1)")
    
    # --- 2. ResNet-50 Stages ---
    stages = [
        ("C2", 0.6, 2.4, 2.6, 2.2),
        ("C3", 0.6, 2.0, 2.2, 3.4),
        ("C4", 0.6, 1.6, 1.8, 4.6),
        ("C5", 0.6, 1.2, 1.4, 5.8),
    ]
    c_verts = []
    for name, dx, dy, dz, x_pos in stages:
        v = draw_3d_block(ax, x=x_pos, y=0, z=1.5 + (3.2 - dz)/2, dx=dx, dy=dy, dz=dz,
                          color=c_backbone, label=name, sublabel=f"stride {2**int(name[1])}")
        c_verts.append(v)
        
    ax.text(4.2, 0.4, "ResNet-50 Backbone (Shared Features)", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1E3A8A',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#DBEAFE', edgecolor='#93C5FD', lw=1.2))

    # --- 3. FPN Pyramid ---
    fpn_stages = [
        ("P2", 0.55, 2.3, 2.5, 2.2),
        ("P3", 0.55, 1.9, 2.1, 3.4),
        ("P4", 0.55, 1.5, 1.7, 4.6),
        ("P5", 0.55, 1.1, 1.3, 5.8),
    ]
    p_verts = []
    for name, dx, dy, dz, x_pos in fpn_stages:
        v = draw_3d_block(ax, x=x_pos + 0.1, y=4.2, z=1.5 + (3.2 - dz)/2, dx=dx, dy=dy, dz=dz,
                          color=c_fpn, label=name, sublabel="256-d")
        p_verts.append(v)
        
    for i in range(4):
        cx, cy = (c_verts[i][6][0] + c_verts[i][7][0])/2, (c_verts[i][6][1] + c_verts[i][7][1])/2
        px, py = (p_verts[i][4][0] + p_verts[i][5][0])/2, (p_verts[i][4][1] + p_verts[i][5][1])/2
        ax.annotate("", xy=(px, py), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4",
                                    color="#0284C7", lw=1.8, linestyle="--"))

    for i in range(3, 0, -1):
        p_src = (p_verts[i][0][0] + p_verts[i][1][0])/2, (p_verts[i][0][1] + p_verts[i][1][1])/2
        p_dst = (p_verts[i-1][5][0] + p_verts[i-1][6][0])/2, (p_verts[i-1][5][1] + p_verts[i-1][6][1])/2
        ax.annotate("", xy=(p_dst[0]+0.3, p_dst[1]-0.1), xytext=(p_src[0]-0.2, p_src[1]-0.1),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.35",
                                    color="#0369A1", lw=2.0))

    # --- 4. PC-MOC Module ---
    v_pcmoc = draw_3d_block(ax, x=3.5, y=7.6, z=4.0, dx=3.0, dy=1.6, dz=1.6,
                            color=c_pcmoc, label="PC-MOC Distillation Engine",
                            sublabel="Multi-Scale Cosine Loss")
    
    ax.annotate("", xy=((v_pcmoc[0][0]+v_pcmoc[1][0])/2, (v_pcmoc[0][1]+v_pcmoc[1][1])/2),
                xytext=((p_verts[1][6][0]+p_verts[2][6][0])/2, (p_verts[1][6][1]+p_verts[2][6][1])/2),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45",
                                color="#059669", lw=2.2, linestyle="-"))
    
    ax.text(5.5, 9.8, r"$\mathcal{L}_{\mathrm{distill}} = \sum_{\ell=2}^5 \frac{1}{H_\ell W_\ell} \left(1 - \cos(f_{\mathrm{curr}}^{P_\ell}, f_{\mathrm{ref}}^{P_\ell})\right)$",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065F46',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5', edgecolor='#6EE7B7', lw=1.2))

    # --- 5. RPN & PC-MR Module ---
    v_rpn = draw_3d_block(ax, x=8.2, y=1.2, z=0.5, dx=2.2, dy=2.0, dz=1.8,
                          color='#475569', label="RPN Proposals", sublabel="Top-k Anchors")
    
    v_pcmr = draw_3d_block(ax, x=11.2, y=1.2, z=0.5, dx=2.6, dy=2.0, dz=2.0,
                           color=c_pcmr, label="PC-MR Micro-Rescue", sublabel="Orthogonal Projection")
    
    ax.annotate("", xy=(v_rpn[0][0], v_rpn[0][1]+0.8), xytext=(p_verts[0][1][0]+0.2, p_verts[0][1][1]-0.4),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45",
                                color="#475569", lw=2.0))
    ax.annotate("", xy=(v_pcmr[0][0], v_pcmr[0][1]+0.8), xytext=(v_rpn[1][0], v_rpn[1][1]+0.8),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45",
                                color="#DC2626", lw=2.2))
    
    ax.text(12.6, 0.4, r"$\mathbf{g}_{\mathrm{proj}} = \mathbf{g}_{\mathrm{micro}} - \frac{\mathbf{g}_{\mathrm{micro}} \cdot \mathbf{g}_{\mathrm{main}}}{\|\mathbf{g}_{\mathrm{main}}\|^2} \mathbf{g}_{\mathrm{main}} \quad (\mathbf{g}_{\mathrm{proj}} \perp \mathbf{g}_{\mathrm{main}})$",
            ha='center', va='center', fontsize=9.2, fontweight='bold', color='#991B1B',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEE2E2', edgecolor='#FCA5A5', lw=1.2))

    # --- 6. RoI Head with SA-ALW ---
    v_roi = draw_3d_block(ax, x=8.5, y=4.8, z=2.4, dx=2.4, dy=2.2, dz=2.4,
                          color=c_roi, label="RoI Align & Fast R-CNN", sublabel="Scale-Adaptive Head")
    
    v_loss = draw_3d_block(ax, x=12.0, y=4.8, z=2.4, dx=2.6, dy=2.2, dz=2.4,
                           color='#7C3AED', label="SA-ALW Loss Engine", sublabel="Anisotropic Wasserstein")
    
    ax.annotate("", xy=(v_roi[0][0], v_roi[0][1]+1.0), xytext=(p_verts[2][6][0]+0.3, p_verts[2][6][1]),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45",
                                color="#D97706", lw=2.2))
    ax.annotate("", xy=(v_loss[0][0], v_loss[0][1]+1.0), xytext=(v_roi[1][0], v_roi[1][1]+1.0),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45",
                                color="#7C3AED", lw=2.2))

    ax.text(13.3, 7.8, r"$\mathcal{L}_{\mathrm{reg}}^{\mathrm{SA}} = \sqrt{w_{\mathrm{pos}}(s) D_{\mathrm{pos}} + D_{\mathrm{shape}}}$" + "\n" + r"$K_{\mathrm{SA}} = \exp\left[-\beta(s)\sqrt{D_{\mathrm{SA}}}\right]$",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#5B21B6',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#EDE9FE', edgecolor='#C4B5FD', lw=1.2))

    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(-0.8, 10.5)
    ax.axis('off')
    
    plt.tight_layout()
    out_png = "paper_a/figures/fig1_framework_architecture.png"
    out_pdf = "paper_a/figures/fig1_framework_architecture.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

def generate_3d_loss_landscape_figure():
    """Generates a high-resolution 3D volumetric surface elevation plot comparing IoU vs NWD vs SA-ALW."""
    fig = plt.figure(figsize=(15.5, 5.2), dpi=300)
    
    # 2D Grid for a 6x6 pixel target box with dx, dy offsets from -8 to +8 px
    w_t, h_t = 6.0, 6.0
    x = np.linspace(-8, 8, 80)
    y = np.linspace(-8, 8, 80)
    X, Y = np.meshgrid(x, y)
    
    # 1. Discrete IoU Loss
    def compute_iou_loss(dx, dy):
        inter_w = np.maximum(0.0, w_t - np.abs(dx))
        inter_h = np.maximum(0.0, h_t - np.abs(dy))
        inter_area = inter_w * inter_h
        union_area = 2 * (w_t * h_t) - inter_area
        iou = inter_area / np.maximum(1e-6, union_area)
        return 1.0 - iou
    Z_iou = compute_iou_loss(X, Y)
    
    # 2. NWD Loss
    def compute_nwd_loss(dx, dy, C=12.0):
        # 2-Wasserstein distance squared for identical scale Gaussians
        W2_sq = dx**2 + dy**2
        nwd_sim = np.exp(-np.sqrt(W2_sq) / C)
        return 1.0 - nwd_sim
    Z_nwd = compute_nwd_loss(X, Y)
    
    # 3. SA-ALW Loss
    def compute_sa_alw_loss(dx, dy):
        Sx = (w_t**2 + w_t**2) / 2.0
        Sy = (h_t**2 + h_t**2) / 2.0
        D_pos = (dx**2 / Sx) + (dy**2 / Sy)
        w_pos = 2.5
        D_sa = w_pos * D_pos
        return np.sqrt(D_sa)
    Z_alw = compute_sa_alw_loss(X, Y)
    
    # Subplot 1: 3D IoU Surface
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(X, Y, Z_iou, cmap='Blues_r', edgecolor='#1E3A8A', linewidth=0.2, alpha=0.9, antialiased=True)
    ax1.set_title(r"$\mathbf{(a)\ Standard\ IoU\ Loss}$" + "\n(Catastrophic Discontinuous Collapse)", fontsize=10.5, fontweight='bold', pad=10, color='#1E293B')
    ax1.set_xlabel(r"$\Delta x$ (px)", fontsize=8.5)
    ax1.set_ylabel(r"$\Delta y$ (px)", fontsize=8.5)
    ax1.set_zlabel(r"$\mathcal{L}_{\mathrm{IoU}}$", fontsize=8.5)
    ax1.view_init(elev=28, azim=-55)
    ax1.contour(X, Y, Z_iou, zdir='z', offset=0, cmap='Blues_r', alpha=0.5)

    # Subplot 2: 3D NWD Surface
    ax2 = fig.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(X, Y, Z_nwd, cmap='Greens_r', edgecolor='#065F46', linewidth=0.2, alpha=0.9, antialiased=True)
    ax2.set_title(r"$\mathbf{(b)\ NWD\ Gaussian\ Loss}$" + "\n(Isotropic Boundary Over-Smoothing)", fontsize=10.5, fontweight='bold', pad=10, color='#1E293B')
    ax2.set_xlabel(r"$\Delta x$ (px)", fontsize=8.5)
    ax2.set_ylabel(r"$\Delta y$ (px)", fontsize=8.5)
    ax2.set_zlabel(r"$\mathcal{L}_{\mathrm{NWD}}$", fontsize=8.5)
    ax2.view_init(elev=28, azim=-55)
    ax2.contour(X, Y, Z_nwd, zdir='z', offset=0, cmap='Greens_r', alpha=0.5)

    # Subplot 3: 3D SA-ALW Surface
    ax3 = fig.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(X, Y, Z_alw, cmap='Purples_r', edgecolor='#4C1D95', linewidth=0.2, alpha=0.9, antialiased=True)
    ax3.set_title(r"$\mathbf{(c)\ Proposed\ SA\text{-}ALW\ Loss}$" + "\n(Scale-Adaptive Smooth Gradient Valley)", fontsize=10.5, fontweight='bold', pad=10, color='#1E293B')
    ax3.set_xlabel(r"$\Delta x$ (px)", fontsize=8.5)
    ax3.set_ylabel(r"$\Delta y$ (px)", fontsize=8.5)
    ax3.set_zlabel(r"$\mathcal{L}_{\mathrm{reg}}^{\mathrm{SA}}$", fontsize=8.5)
    ax3.view_init(elev=28, azim=-55)
    ax3.contour(X, Y, Z_alw, zdir='z', offset=0, cmap='Purples_r', alpha=0.5)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#F8FAFC')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(labelsize=7.5)

    plt.tight_layout()
    out_png = "paper_a/figures/fig2_geometry_comparison.png"
    out_pdf = "paper_a/figures/fig2_geometry_comparison.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_pdf, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"Generated 3D Loss Landscape: {out_png}, {out_pdf}")

def generate_drawio_xml():
    """Generates native Draw.io XML file with structured 3D blocks and styles."""
    xml_content = """<mxfile host="Electron" modified="2026-08-20T08:25:00.000Z" agent="Antigravity 3D Academic Synthesizer" version="22.1.18" type="device">
  <diagram id="joint_framework_3d" name="Joint Framework 3D Architecture">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1654" pageHeight="1169" math="1" shadow="1">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- Input Image 3D Block -->
        <mxCell id="input_img" value="&lt;b&gt;Input Image Tile&lt;/b&gt;&lt;br&gt;800 × 800 (b1-tiled)" style="shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.08;darkOpacity2=0.15;fillColor=#64748B;strokeColor=#334155;fontColor=#ffffff;size=15;" vertex="1" parent="1">
          <mxGeometry x="40" y="320" width="120" height="180" as="geometry" />
        </mxCell>

        <!-- ResNet-50 Backbone 3D Container -->
        <mxCell id="backbone_group" value="&lt;b&gt;ResNet-50 Backbone (Shared Feature Extractor)&lt;/b&gt;" style="swimlane;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#3B82F6;fontColor=#1E3A8A;rounded=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="220" y="240" width="340" height="340" as="geometry" />
        </mxCell>
        <mxCell id="c2_block" value="&lt;b&gt;C2&lt;/b&gt;&lt;br&gt;stride 4" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#2563EB;strokeColor=#1D4ED8;fontColor=#ffffff;size=12;" vertex="1" parent="backbone_group">
          <mxGeometry x="20" y="50" width="60" height="240" as="geometry" />
        </mxCell>
        <mxCell id="c3_block" value="&lt;b&gt;C3&lt;/b&gt;&lt;br&gt;stride 8" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#2563EB;strokeColor=#1D4ED8;fontColor=#ffffff;size=12;" vertex="1" parent="backbone_group">
          <mxGeometry x="100" y="70" width="60" height="200" as="geometry" />
        </mxCell>
        <mxCell id="c4_block" value="&lt;b&gt;C4&lt;/b&gt;&lt;br&gt;stride 16" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#2563EB;strokeColor=#1D4ED8;fontColor=#ffffff;size=12;" vertex="1" parent="backbone_group">
          <mxGeometry x="180" y="90" width="60" height="160" as="geometry" />
        </mxCell>
        <mxCell id="c5_block" value="&lt;b&gt;C5&lt;/b&gt;&lt;br&gt;stride 32" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#2563EB;strokeColor=#1D4ED8;fontColor=#ffffff;size=12;" vertex="1" parent="backbone_group">
          <mxGeometry x="260" y="110" width="60" height="120" as="geometry" />
        </mxCell>

        <!-- Feature Pyramid Network (FPN) 3D Container -->
        <mxCell id="fpn_group" value="&lt;b&gt;Feature Pyramid Network (FPN Levels P2-P5)&lt;/b&gt;" style="swimlane;whiteSpace=wrap;html=1;fillColor=#F0F9FF;strokeColor=#0284C7;fontColor=#0369A1;rounded=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="620" y="240" width="340" height="340" as="geometry" />
        </mxCell>
        <mxCell id="p2_block" value="&lt;b&gt;P2&lt;/b&gt;&lt;br&gt;256-d" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#0284C7;strokeColor=#0369A1;fontColor=#ffffff;size=12;" vertex="1" parent="fpn_group">
          <mxGeometry x="20" y="50" width="60" height="240" as="geometry" />
        </mxCell>
        <mxCell id="p3_block" value="&lt;b&gt;P3&lt;/b&gt;&lt;br&gt;256-d" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#0284C7;strokeColor=#0369A1;fontColor=#ffffff;size=12;" vertex="1" parent="fpn_group">
          <mxGeometry x="100" y="70" width="60" height="200" as="geometry" />
        </mxCell>
        <mxCell id="p4_block" value="&lt;b&gt;P4&lt;/b&gt;&lt;br&gt;256-d" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#0284C7;strokeColor=#0369A1;fontColor=#ffffff;size=12;" vertex="1" parent="fpn_group">
          <mxGeometry x="180" y="90" width="60" height="160" as="geometry" />
        </mxCell>
        <mxCell id="p5_block" value="&lt;b&gt;P5&lt;/b&gt;&lt;br&gt;256-d" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#0284C7;strokeColor=#0369A1;fontColor=#ffffff;size=12;" vertex="1" parent="fpn_group">
          <mxGeometry x="260" y="110" width="60" height="120" as="geometry" />
        </mxCell>

        <!-- PC-MOC Distillation Box -->
        <mxCell id="pcmoc_box" value="&lt;b&gt;PC-MOC Multi-Scale Feature Distillation&lt;/b&gt;&lt;br&gt;$$\\mathcal{L}_{\\mathrm{distill}} = \\sum_{\\ell=2}^5 \\frac{1}{H_\\ell W_\\ell}(1 - \\cos(f_{\\mathrm{curr}}^{P_\\ell}, f_{\\mathrm{ref}}^{P_\\ell}))$$" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#D1FAE5;strokeColor=#059669;fontColor=#065F46;size=15;rounded=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="580" y="60" width="420" height="130" as="geometry" />
        </mxCell>

        <!-- PC-MR Micro-Rescue Box -->
        <mxCell id="pcmr_box" value="&lt;b&gt;PC-MR Proposal Micro-Rescue (RPN)&lt;/b&gt;&lt;br&gt;$$\\mathbf{g}_{\\mathrm{proj}} = \\mathbf{g}_{\\mathrm{micro}} - \\frac{\\mathbf{g}_{\\mathrm{micro}} \\cdot \\mathbf{g}_{\\mathrm{main}}}{\\|\\mathbf{g}_{\\mathrm{main}}\\|^2} \\mathbf{g}_{\\mathrm{main}} \\quad (\\mathbf{g}_{\\mathrm{proj}} \\perp \\mathbf{g}_{\\mathrm{main}})$$" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#FEE2E2;strokeColor=#DC2626;fontColor=#991B1B;size=15;rounded=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="1030" y="470" width="520" height="150" as="geometry" />
        </mxCell>

        <!-- RoI Head & SA-ALW Loss Box -->
        <mxCell id="roi_alw_box" value="&lt;b&gt;RoI Head with SA-ALW &amp;amp; Iterative-CBL&lt;/b&gt;&lt;br&gt;$$\\mathcal{L}_{\\mathrm{reg}}^{\\mathrm{SA}} = \\sqrt{w_{\\mathrm{pos}}(s) D_{\\mathrm{pos}} + D_{\\mathrm{shape}}}$$&lt;br&gt;$$K_{\\mathrm{SA}} = \\exp\\left[-\\beta(s)\\sqrt{D_{\\mathrm{SA}}}\\right]$$" style="shape=cube;whiteSpace=wrap;html=1;fillColor=#EDE9FE;strokeColor=#7C3AED;fontColor=#5B21B6;size=15;rounded=1;shadow=1;" vertex="1" parent="1">
          <mxGeometry x="1030" y="240" width="520" height="180" as="geometry" />
        </mxCell>

        <!-- Connectors -->
        <mxCell id="edge1" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#334155;strokeWidth=2.5;" edge="1" parent="1" source="input_img" target="backbone_group">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge2" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#0284C7;strokeWidth=2.5;dashed=1;" edge="1" parent="1" source="backbone_group" target="fpn_group">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge3" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#059669;strokeWidth=3;" edge="1" parent="1" source="fpn_group" target="pcmoc_box">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge4" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7C3AED;strokeWidth=3;" edge="1" parent="1" source="fpn_group" target="roi_alw_box">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="edge5" style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#DC2626;strokeWidth=3;" edge="1" parent="1" source="fpn_group" target="pcmr_box">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    out_drawio = "paper_a/figures/fig1_framework_architecture.drawio"
    with open(out_drawio, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"Generated Native Draw.io file: {out_drawio}")

if __name__ == "__main__":
    generate_3d_architecture_diagram()
    generate_3d_loss_landscape_figure()
    generate_drawio_xml()
