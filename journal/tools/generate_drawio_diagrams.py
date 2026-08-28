"""
Generate standard, valid, editable .drawio XML diagram files for H-WIoU paper.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
FIG_DIR = ROOT / "journal/figures"
MANUSCRIPT_FIG_DIR = ROOT / "journal/manuscript/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_FIG_DIR.mkdir(parents=True, exist_ok=True)

def create_pipeline_architecture_drawio():
    xml_content = """<mxfile host="Electron" modified="2026-08-26T00:00:00.000Z" agent="Antigravity" version="21.6.8" type="device">
  <diagram id="hwiou_pipeline" name="H-WIoU Pipeline Architecture">
    <mxGraphModel dx="1422" dy="794" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1300" pageHeight="650" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- MAIN TITLE BANNER -->
        <mxCell id="title" value="&lt;b&gt;&lt;font style=&quot;font-size: 18px;&quot;&gt;H-WIoU: End-to-End Scale-Aware Homotopy Detection Pipeline for Tiny Objects&lt;/font&gt;&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#64748b&quot; style=&quot;font-size: 12px;&quot;&gt;Two-Stage Architecture with Continuous Homotopy RPN Soft-Assignment &amp;amp; Bounded RoI Regression&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#E2E8F0;arcSize=6;" vertex="1" parent="1">
          <mxGeometry x="40" y="30" width="1220" height="50" as="geometry" />
        </mxCell>

        <!-- CONTAINER 1: BACKBONE & FPN -->
        <mxCell id="c1" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F8FAFC;strokeColor=#CBD5E1;strokeWidth=1.5;arcSize=6;" vertex="1" parent="1">
          <mxGeometry x="40" y="100" width="280" height="500" as="geometry" />
        </mxCell>
        <mxCell id="c1_header" value="&lt;b&gt;Stage 1: Multi-Scale Backbone&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#1E293B;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="55" y="115" width="250" height="32" as="geometry" />
        </mxCell>

        <!-- Input Image -->
        <mxCell id="input_img" value="&lt;b&gt;Input Aerial Image&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#475569&quot;&gt;1024 × 1024 px&lt;br&gt;(Tiny Objects &amp;lt; 16px)&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#94A3B8;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="75" y="170" width="210" height="60" as="geometry" />
        </mxCell>

        <!-- ResNet-50 -->
        <mxCell id="resnet" value="&lt;b&gt;ResNet-50 Backbone&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#475569&quot;&gt;C2, C3, C4, C5 Stages&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#3B82F6;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="75" y="260" width="210" height="55" as="geometry" />
        </mxCell>

        <!-- FPN Pyramid -->
        <mxCell id="fpn_p2" value="&lt;b&gt;FPN P2&lt;/b&gt; (Stride 4, 256×256)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="75" y="345" width="210" height="35" as="geometry" />
        </mxCell>
        <mxCell id="fpn_p3" value="&lt;b&gt;FPN P3&lt;/b&gt; (Stride 8, 128×128)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#BFDBFE;strokeColor=#2563EB;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="75" y="390" width="210" height="35" as="geometry" />
        </mxCell>
        <mxCell id="fpn_p4" value="&lt;b&gt;FPN P4&lt;/b&gt; (Stride 16, 64×64)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#93C5FD;strokeColor=#1D4ED8;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="75" y="435" width="210" height="35" as="geometry" />
        </mxCell>
        <mxCell id="fpn_p5" value="&lt;b&gt;FPN P5&lt;/b&gt; (Stride 32, 32×32)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#60A5FA;strokeColor=#1E40AF;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="75" y="480" width="210" height="35" as="geometry" />
        </mxCell>

        <mxCell id="fpn_note" value="&lt;font color=&quot;#1e40af&quot; style=&quot;font-size: 11px;&quot;&gt;Lateral 1×1 Conv + 2× Upsampling&lt;/font&gt;" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;" vertex="1" parent="1">
          <mxGeometry x="80" y="535" width="200" height="30" as="geometry" />
        </mxCell>


        <!-- CONTAINER 2: DYNAMIC HOMOTOPY RPN ASSIGNMENT (PROPOSED) -->
        <mxCell id="c2" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF7ED;strokeColor=#EA580C;strokeWidth=2;arcSize=6;" vertex="1" parent="1">
          <mxGeometry x="360" y="100" width="460" height="500" as="geometry" />
        </mxCell>
        <mxCell id="c2_header" value="&lt;b&gt;Stage 2: Scale-Aware Homotopy Assignment (RPN)&lt;/b&gt; &lt;font color=&quot;#fed7aa&quot;&gt;[PROPOSED]&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EA580C;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="380" y="115" width="420" height="32" as="geometry" />
        </mxCell>

        <!-- Kernel 1: Dynamic Weighting -->
        <mxCell id="gamma_box" value="&lt;b&gt;1. Dynamic Scale-Dependent Parameter &amp;gamma;(d)&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size: 13px;&quot; color=&quot;#9a3412&quot;&gt;$$\gamma(d) = \frac{1}{1 + (d / \sigma_0)^2}, \quad d = \sqrt{w \cdot h}$$&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#7c2d12&quot; style=&quot;font-size: 11px;&quot;&gt;Rational Cauchy Weighting (Characteristic Scale &amp;sigma;&lt;sub&gt;0&lt;/sub&gt; = 8.0 px)&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#FB923C;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="390" y="165" width="400" height="80" as="geometry" />
        </mxCell>

        <!-- Kernel 2: Continuous Convex Combination -->
        <mxCell id="homotopy_box" value="&lt;b&gt;2. Continuous Homotopy Metric Space&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size: 13px;&quot; color=&quot;#9a3412&quot;&gt;$$H_\gamma(A, G) = (1 - \gamma(d)) \cdot \mathrm{IoU}(A, G) + \gamma(d) \cdot \mathrm{NWD}(A, G)$$&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#7c2d12&quot; style=&quot;font-size: 11px;&quot;&gt;Seamless topological deformation between Wasserstein transport &amp;amp; Jaccard overlap&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#FB923C;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="390" y="265" width="400" height="85" as="geometry" />
        </mxCell>

        <!-- Regime Properties -->
        <mxCell id="regimes" value="&lt;table style=&quot;width:100%;font-size:11px;&quot;&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;• Tiny Regime (d &amp;lt; 8px):&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&amp;gamma; &amp;rarr; 1 &amp;implies; H&lt;sub&gt;&amp;gamma;&lt;/sub&gt; &amp;rarr; NWD (Zero IoU Collapse Prevented)&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;• Normal Regime (d &amp;gt; 32px):&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&amp;gamma; &amp;rarr; 0 &amp;implies; H&lt;sub&gt;&amp;gamma;&lt;/sub&gt; &amp;rarr; IoU (Standard Geometry Preserved)&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEDD5;strokeColor=#FDBA74;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="390" y="365" width="400" height="55" as="geometry" />
        </mxCell>

        <!-- RPN Soft Target Assignment -->
        <mxCell id="rpn_assignment" value="&lt;b&gt;3. Top-k Soft Label Candidate Assigner&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#431407&quot; style=&quot;font-size: 11px;&quot;&gt;Assigns positive anchors via ranking on H&lt;sub&gt;&amp;gamma;&lt;/sub&gt;(A, G) matrix.&lt;br&gt;Provides smooth, non-vanishing gradients &amp;nabla;&lt;sub&gt;A&lt;/sub&gt; H&lt;sub&gt;&amp;gamma;&lt;/sub&gt; &amp;ne; 0 everywhere.&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#EA580C;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="390" y="435" width="400" height="75" as="geometry" />
        </mxCell>

        <mxCell id="rpn_out" value="&lt;b&gt;High-Recall Tiny RoI Proposals (Top 2000)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EA580C;strokeColor=none;fontColor=#FFFFFF;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="460" y="530" width="260" height="35" as="geometry" />
        </mxCell>


        <!-- CONTAINER 3: FAST R-CNN HEAD & BOUNDED REGRESSION -->
        <mxCell id="c3" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#16A34A;strokeWidth=1.5;arcSize=6;" vertex="1" parent="1">
          <mxGeometry x="860" y="100" width="400" height="500" as="geometry" />
        </mxCell>
        <mxCell id="c3_header" value="&lt;b&gt;Stage 3: Fast R-CNN &amp;amp; Bounded Loss&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#166534;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="880" y="115" width="360" height="32" as="geometry" />
        </mxCell>

        <!-- RoIAlign -->
        <mxCell id="roialign" value="&lt;b&gt;RoIAlign Feature Extraction&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#14532d&quot; style=&quot;font-size: 11px;&quot;&gt;7 × 7 Bilinear Interpolation from Selected FPN Level P&lt;sub&gt;k&lt;/sub&gt;&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#4ADE80;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="890" y="165" width="340" height="55" as="geometry" />
        </mxCell>

        <!-- 2-Layer MLP -->
        <mxCell id="mlp" value="&lt;b&gt;Two-Layer MLP Head&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#14532d&quot; style=&quot;font-size: 11px;&quot;&gt;2 × FC (1024-d) with ReLU Activation&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#DCFCE7;strokeColor=#22C55E;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="890" y="240" width="340" height="50" as="geometry" />
        </mxCell>

        <!-- Dual Branches -->
        <mxCell id="cls_branch" value="&lt;b&gt;Classification Branch&lt;/b&gt;&lt;br&gt;$$\mathcal{L}_{\mathrm{cls}} = \mathrm{CrossEntropy}(p, y)$$" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#86EFAC;strokeWidth=1;" vertex="1" parent="1">
          <mxGeometry x="890" y="310" width="340" height="60" as="geometry" />
        </mxCell>

        <mxCell id="reg_branch" value="&lt;b&gt;Bounded Homotopy Regression Branch&lt;/b&gt;&lt;br&gt;&lt;font color=&quot;#15803d&quot; style=&quot;font-size: 13px;&quot;&gt;$$\mathcal{L}_{\mathrm{H\text{-}WIoU}} = 1 - H_\gamma(B_{\mathrm{pred}}, B_{\mathrm{gt}})$$&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#166534&quot; style=&quot;font-size: 11px;&quot;&gt;Strictly Bounded in [0, 1] — Prevents Gradient Explosion on Sub-pixel Errors&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#16A34A;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="890" y="390" width="340" height="95" as="geometry" />
        </mxCell>

        <!-- Final Detections Output -->
        <mxCell id="final_det" value="&lt;b&gt;Calibrated Final Tiny Detections&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size: 12px;&quot;&gt;&lt;b&gt;AI-TOD-v2 mAP&lt;sub&gt;50&lt;/sub&gt;: 46.2% (+19.9%)&lt;/b&gt; | &lt;b&gt;Fair-20 AP&lt;sub&gt;50&lt;/sub&gt;: 46.34%&lt;/b&gt;&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#166534;strokeColor=none;fontColor=#FFFFFF;" vertex="1" parent="1">
          <mxGeometry x="890" y="510" width="340" height="55" as="geometry" />
        </mxCell>


        <!-- CONNECTORS / ARROWS -->
        <mxCell id="a1" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#475569;strokeWidth=2;entryX=0.5;entryY=0;" edge="1" parent="1" source="input_img" target="resnet">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a2" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#475569;strokeWidth=2;entryX=0.5;entryY=0;" edge="1" parent="1" source="resnet" target="fpn_p2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a3" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#2563EB;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="fpn_p3" target="gamma_box">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a4" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#EA580C;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="gamma_box" target="homotopy_box">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a5" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#EA580C;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="homotopy_box" target="regimes">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a6" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#EA580C;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="regimes" target="rpn_assignment">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a7" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#EA580C;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="rpn_assignment" target="rpn_out">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a8" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="rpn_out" target="roialign">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a9" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="roialign" target="mlp">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a10" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="mlp" target="cls_branch">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a11" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="mlp" target="reg_branch">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="a12" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;" edge="1" parent="1" source="reg_branch" target="final_det">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    out_file1 = FIG_DIR / "hwiou_pipeline_architecture.drawio"
    out_file2 = MANUSCRIPT_FIG_DIR / "fig5_pipeline_architecture.drawio"
    out_file1.write_text(xml_content.strip(), encoding="utf-8")
    out_file2.write_text(xml_content.strip(), encoding="utf-8")
    print(f"Generated drawio pipeline architecture at: {out_file1} and {out_file2}")

def create_homotopy_theory_drawio():
    xml_content = r"""<mxfile host="Electron" modified="2026-08-26T00:00:00.000Z" agent="Antigravity" version="21.6.8" type="device">
  <diagram id="hwiou_theory" name="Homotopy Metric Theory">
    <mxGraphModel dx="1200" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="500" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- CARD 1: STANDARD IOU COLLAPSE -->
        <mxCell id="c1" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#FECACA;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="40" y="50" width="300" height="380" as="geometry" />
        </mxCell>
        <mxCell id="c1_h" value="&lt;b&gt;(a) Standard IoU Collapse&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#991B1B;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="60" y="65" width="260" height="30" as="geometry" />
        </mxCell>
        <mxCell id="box_a" value="Anchor A (8×8)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FCA5A5;strokeColor=#DC2626;fontColor=#7F1D1D;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="70" y="120" width="90" height="70" as="geometry" />
        </mxCell>
        <mxCell id="box_g" value="GT G (6×6)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#93C5FD;strokeColor=#2563EB;fontColor=#1E3A8A;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="220" y="130" width="80" height="60" as="geometry" />
        </mxCell>
        <mxCell id="iou_desc" value="&lt;b&gt;Area(A &amp;cap; G) = 0&lt;/b&gt;&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#991b1b&quot; style=&quot;font-size:13px;&quot;&gt;$$\mathrm{IoU}(A, G) = 0$$&lt;br&gt;$$\nabla_A \mathrm{IoU} = \mathbf{0}$$&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#7f1d1d&quot;&gt;Gradient completely vanishes when boxes do not overlap at micro-scales.&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#FECACA;" vertex="1" parent="1">
          <mxGeometry x="60" y="220" width="260" height="180" as="geometry" />
        </mxCell>

        <!-- CARD 2: GAUSSIAN WASSERSTEIN TRANSPORT -->
        <mxCell id="c2" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#BBF7D0;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="380" y="50" width="320" height="380" as="geometry" />
        </mxCell>
        <mxCell id="c2_h" value="&lt;b&gt;(b) Gaussian Wasserstein Space&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#166534;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="400" y="65" width="280" height="30" as="geometry" />
        </mxCell>
        <mxCell id="gauss_a" value="&amp;Nu;&lt;sub&gt;A&lt;/sub&gt;(&amp;mu;&lt;sub&gt;A&lt;/sub&gt;, &amp;Sigma;&lt;sub&gt;A&lt;/sub&gt;)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#86EFAC;strokeColor=#16A34A;fontColor=#14532D;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="410" y="120" width="100" height="70" as="geometry" />
        </mxCell>
        <mxCell id="gauss_g" value="&amp;Nu;&lt;sub&gt;G&lt;/sub&gt;(&amp;mu;&lt;sub&gt;G&lt;/sub&gt;, &amp;Sigma;&lt;sub&gt;G&lt;/sub&gt;)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#93C5FD;strokeColor=#2563EB;fontColor=#1E3A8A;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="580" y="125" width="90" height="60" as="geometry" />
        </mxCell>
        <mxCell id="trans_arr" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#16A34A;strokeWidth=2;dashed=1;entryX=0;entryY=0.5;exitX=1;exitY=0.5;" edge="1" parent="1" source="gauss_a" target="gauss_g">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="nwd_desc" value="&lt;b&gt;Optimal Transport Distance&lt;/b&gt;&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#166534&quot; style=&quot;font-size:12px;&quot;&gt;$$W_2^2(\mathcal{N}_A, \mathcal{N}_G) = \|\mu_A - \mu_G\|_2^2 + \dots$$&lt;br&gt;$$\mathrm{NWD} = \exp\left(-\frac{W_2(A, G)}{C}\right) &gt; 0$$&lt;/font&gt;&lt;br&gt;&lt;font color=&quot;#14532d&quot;&gt;Strictly positive and continuous everywhere across $\mathbb{R}^2$.&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#BBF7D0;" vertex="1" parent="1">
          <mxGeometry x="400" y="220" width="280" height="180" as="geometry" />
        </mxCell>

        <!-- CARD 3: CONTINUOUS HOMOTOPY CONVEX COMBINATION (PROPOSED) -->
        <mxCell id="c3" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF7ED;strokeColor=#EA580C;strokeWidth=2;" vertex="1" parent="1">
          <mxGeometry x="740" y="50" width="320" height="380" as="geometry" />
        </mxCell>
        <mxCell id="c3_h" value="&lt;b&gt;(c) Continuous Homotopy H-WIoU&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#EA580C;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="760" y="65" width="280" height="30" as="geometry" />
        </mxCell>
        <mxCell id="homo_form" value="&lt;b&gt;Continuous Homotopy Combination&lt;/b&gt;&lt;br&gt;&lt;br&gt;&lt;font color=&quot;#c2410c&quot; style=&quot;font-size:12px;&quot;&gt;$$H_\gamma(A, G) = (1 - \gamma)\mathrm{IoU} + \gamma \mathrm{NWD}$$&lt;br&gt;$$\gamma(d) = \frac{1}{1 + (d / \sigma_0)^2}$$&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#FDBA74;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="760" y="115" width="280" height="100" as="geometry" />
        </mxCell>
        <mxCell id="homo_desc" value="&lt;table style=&quot;width:100%;font-size:11px;&quot;&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;• d &amp;rarr; 0 (Tiny):&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&amp;gamma; &amp;rarr; 1 &amp;implies; H&lt;sub&gt;&amp;gamma;&lt;/sub&gt; &amp;rarr; NWD&lt;/td&gt;&lt;/tr&gt;&lt;tr&gt;&lt;td&gt;&lt;b&gt;• d &amp;rarr; &amp;infin; (Normal):&lt;/b&gt;&lt;/td&gt;&lt;td&gt;&amp;gamma; &amp;rarr; 0 &amp;implies; H&lt;sub&gt;&amp;gamma;&lt;/sub&gt; &amp;rarr; IoU&lt;/td&gt;&lt;/tr&gt;&lt;/table&gt;&lt;br&gt;&lt;font color=&quot;#9a3412&quot; style=&quot;font-size:11px;&quot;&gt;&lt;b&gt;Smooth topological deformation guaranteeing non-vanishing gradients across all object scales.&lt;/b&gt;&lt;/font&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEDD5;strokeColor=#FDBA74;" vertex="1" parent="1">
          <mxGeometry x="760" y="235" width="280" height="165" as="geometry" />
        </mxCell>

        <!-- CONNECTORS -->
        <mxCell id="conn1" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#64748B;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="c1" target="c2">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>
        <mxCell id="conn2" style="edgeStyle=orthogonalEdgeStyle;rounded=1;strokeColor=#EA580C;strokeWidth=2;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="c2" target="c3">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    out_file1 = FIG_DIR / "hwiou_homotopy_theory.drawio"
    out_file2 = MANUSCRIPT_FIG_DIR / "fig1_homotopy_theory.drawio"
    out_file1.write_text(xml_content.strip(), encoding="utf-8")
    out_file2.write_text(xml_content.strip(), encoding="utf-8")
    print(f"Generated drawio homotopy theory at: {out_file1} and {out_file2}")

def main():
    create_pipeline_architecture_drawio()
    create_homotopy_theory_drawio()


if __name__ == "__main__":
    main()
