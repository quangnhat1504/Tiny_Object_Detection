"""
Generate editable .drawio source files for Figure 1 (Teaser) and Figure 2 (Math Intuition).
"""
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
FIG_DIR = ROOT / "journal/manuscript/figures"

def generate_fig1_teaser_drawio():
    xml = r"""<mxfile host="Electron" modified="2026-08-26T00:00:00.000Z" agent="Antigravity" version="21.6.8" type="device">
  <diagram id="fig1_teaser" name="Figure 1 Teaser Motivation">
    <mxGraphModel dx="1200" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1100" pageHeight="500" math="1" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- LEFT PANEL: STANDARD RPN STARVATION -->
        <mxCell id="p_left" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FEF2F2;strokeColor=#FECACA;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="40" y="40" width="480" height="420" as="geometry" />
        </mxCell>
        <mxCell id="p_left_title" value="&lt;b&gt;Standard RPN: Positive Anchor Starvation (&amp;gt;70%)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#991B1B;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="60" y="55" width="440" height="30" as="geometry" />
        </mxCell>

        <!-- Tiny GT Box -->
        <mxCell id="gt_left" value="&lt;b&gt;Tiny GT&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;&quot;&gt;s &amp;lt; 8px&lt;/font&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#93C5FD;strokeColor=#1E3A8A;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="250" y="190" width="50" height="50" as="geometry" />
        </mxCell>

        <!-- Disjoint Anchor -->
        <mxCell id="anchor_left" value="&lt;b&gt;Anchor A&lt;sub&gt;i&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;&quot;&gt;IoU &amp;lt; 0.2&lt;/font&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FCA5A5;strokeColor=#DC2626;strokeWidth=1.5;dashed=1;" vertex="1" parent="1">
          <mxGeometry x="320" y="160" width="100" height="100" as="geometry" />
        </mxCell>

        <mxCell id="gap_left" value="&lt;b&gt;Zero Overlap (Area &amp;cap; = 0)&lt;/b&gt;&lt;br&gt;$$\nabla_A \mathcal{L}_{\mathrm{IoU}} = \mathbf{0}$$" style="text;html=1;align=center;verticalAlign=middle;fontSize=11;fontColor=#991B1B;" vertex="1" parent="1">
          <mxGeometry x="250" y="270" width="180" height="40" as="geometry" />
        </mxCell>

        <!-- Left Stats Banner -->
        <mxCell id="stats_left" value="&lt;b&gt;Catastrophic Failure on Micro Objects:&lt;/b&gt;&lt;br&gt;• &lt;b&gt;Positive Survival Rate:&lt;/b&gt; &lt;font color=&quot;#991b1b&quot;&gt;&lt;b&gt;0.18&lt;/b&gt;&lt;/font&gt; (82% Target Loss)&lt;br&gt;• &lt;b&gt;Gradient Vanishing:&lt;/b&gt; Zero optimization feedback on sub-pixel shifts" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#FECACA;fontSize=11;align=left;spacingLeft=10;" vertex="1" parent="1">
          <mxGeometry x="60" y="340" width="440" height="90" as="geometry" />
        </mxCell>


        <!-- RIGHT PANEL: H-WIoU STAGE 1 (HLA) -->
        <mxCell id="p_right" value="" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F0FDF4;strokeColor=#BBF7D0;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="560" y="40" width="480" height="420" as="geometry" />
        </mxCell>
        <mxCell id="p_right_title" value="&lt;b&gt;H-WIoU Stage 1: Homotopy Assignment (Survival 0.94)&lt;/b&gt;" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#166534;strokeColor=none;fontColor=#FFFFFF;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="580" y="55" width="440" height="30" as="geometry" />
        </mxCell>

        <!-- Gaussian Field -->
        <mxCell id="gauss_field" value="" style="ellipse;whiteSpace=wrap;html=1;fillColor=#6EE7B7;strokeColor=#0F766E;strokeWidth=1.5;opacity=40;dashed=1;" vertex="1" parent="1">
          <mxGeometry x="700" y="120" width="200" height="200" as="geometry" />
        </mxCell>

        <!-- Tiny GT Box -->
        <mxCell id="gt_right" value="&lt;b&gt;Tiny GT&lt;/b&gt;&lt;br&gt;&lt;font style=&quot;font-size:10px;&quot;&gt;s &amp;lt; 8px&lt;/font&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#93C5FD;strokeColor=#1E3A8A;strokeWidth=1.5;" vertex="1" parent="1">
          <mxGeometry x="775" y="195" width="50" height="50" as="geometry" />
        </mxCell>

        <!-- Survived Positive Anchors -->
        <mxCell id="pos1" value="A&lt;sub&gt;1&lt;/sub&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#A7F3D0;strokeColor=#0F766E;strokeWidth=1;opacity=80;" vertex="1" parent="1">
          <mxGeometry x="740" y="160" width="50" height="50" as="geometry" />
        </mxCell>
        <mxCell id="pos2" value="A&lt;sub&gt;2&lt;/sub&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#A7F3D0;strokeColor=#0F766E;strokeWidth=1;opacity=80;" vertex="1" parent="1">
          <mxGeometry x="810" y="160" width="50" height="50" as="geometry" />
        </mxCell>
        <mxCell id="pos3" value="A&lt;sub&gt;3&lt;/sub&gt;" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#A7F3D0;strokeColor=#0F766E;strokeWidth=1;opacity=80;" vertex="1" parent="1">
          <mxGeometry x="760" y="230" width="50" height="50" as="geometry" />
        </mxCell>

        <!-- Right Stats Banner -->
        <mxCell id="stats_right" value="&lt;b&gt;Homotopy Label Assignment (HLA) Breakthrough:&lt;/b&gt;&lt;br&gt;• &lt;b&gt;Positive Survival Rate:&lt;/b&gt; &lt;font color=&quot;#166534&quot;&gt;&lt;b&gt;0.18 &amp;rarr; 0.94&lt;/b&gt;&lt;/font&gt; (+422% Gain)&lt;br&gt;• &lt;b&gt;Gaussian Receptive Field:&lt;/b&gt; Smooth optimal transport &amp;nabla;&lt;sub&gt;A&lt;/sub&gt; &amp;ne; 0 everywhere" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#BBF7D0;fontSize=11;align=left;spacingLeft=10;" vertex="1" parent="1">
          <mxGeometry x="580" y="340" width="440" height="90" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""
    (FIG_DIR / "fig1_homotopy_theory.drawio").write_text(xml.strip(), encoding="utf-8")
    print("Saved fig1_homotopy_theory.drawio")

if __name__ == "__main__":
    generate_fig1_teaser_drawio()
