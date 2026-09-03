"""
Compile H-WIoU Architecture Diagram using Research Diagram Architect 3.0.
Generates:
1. journal/figures/hwiou_pipeline_architecture.drawio (Native Draw.io XML with MathJax math="1")
2. journal/figures/hwiou_pipeline_architecture.tex (Standalone LaTeX TikZ)
3. journal/figures/hwiou_pipeline_architecture.svg (Responsive Vector SVG)
4. journal/figures/fig5_pipeline_architecture.pdf & .png (Clean, basic, modular block diagram)
"""
import sys
import os
from pathlib import Path

# Add research-diagram-architect to path
RDA_PATH = r"C:\Users\ADMIN\.gemini\config\skills\research-diagram-architect"
if RDA_PATH not in sys.path:
    sys.path.insert(0, RDA_PATH)

import research_diagram_architect as rda
from research_diagram_architect import (
    UniversalDiagramSpec, Container, Node, Edge,
    UnifiedStyleSpec, compute_layout, render
)

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
FIG_DIR = ROOT / "journal/figures"
FIG_MANUSCRIPT_DIR = ROOT / "journal/manuscript/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def build_rda_spec() -> UniversalDiagramSpec:
    """Build a clean, modular UniversalDiagramSpec for H-WIoU."""
    containers = [
        Container(id="backbone_stage", label="1. Multi-Scale Feature Extraction", order=0),
        Container(id="rpn_stage", label="2. Stage 1: RPN Homotopy Label Assignment", order=1),
        Container(id="homotopy_core", label="3. Theoretical Scale-Homotopy Foundation", order=2),
        Container(id="roi_stage", label="4. Stage 2: RoI Head & Multi-Task Loss", order=3),
    ]

    nodes = [
        # Container 1: Backbone & FPN
        Node(
            id="input_img",
            label="Input Aerial Image",
            container_id="backbone_stage",
            kind="dataset",
            latex_math=r"I \in \mathbb{R}^{H \times W \times 3}",
            order=0
        ),
        Node(
            id="resnet50",
            label="ResNet-50 Backbone",
            container_id="backbone_stage",
            kind="process",
            latex_math=r"C_2, C_3, C_4, C_5",
            order=1
        ),
        Node(
            id="fpn",
            label="Feature Pyramid Network",
            container_id="backbone_stage",
            kind="tensor_3d",
            latex_math=r"P_2, P_3, P_4, P_5",
            order=2,
            emphasize=True
        ),

        # Container 2: RPN & HLA
        Node(
            id="anchor_gen",
            label="Dense Anchor Generation",
            container_id="rpn_stage",
            kind="process",
            latex_math=r"\{A_i\}_{i=1}^N",
            order=0
        ),
        Node(
            id="hla_module",
            label="Homotopy Label Assignment (HLA)",
            container_id="rpn_stage",
            kind="highlight",
            latex_math=r"\mathbf{S}_{ij} = \mathcal{S}_{\text{H-WIoU}}(A_i, G_j)",
            order=1,
            emphasize=True
        ),
        Node(
            id="rpn_proposals",
            label="Region Proposals (RoIs)",
            container_id="rpn_stage",
            kind="pill_op",
            latex_math=r"\mathbf{R} = \{\mathbf{r}_k\}_{k=1}^K",
            order=2
        ),

        # Container 3: Theoretical Core
        Node(
            id="homotopy_metric",
            label="Unified Homotopy Metric",
            container_id="homotopy_core",
            kind="model",
            latex_math=r"\mathcal{S}_{\text{H-WIoU}} = [\text{IoU}]^{\gamma(s)} \cdot e^{-(1-\gamma(s))\mathcal{D}_{\mathcal{W}}^2}",
            order=0,
            emphasize=True
        ),
        Node(
            id="scale_weight",
            label="Scale Transition Function",
            container_id="homotopy_core",
            kind="process",
            latex_math=r"\gamma(s) = \frac{s^2}{s^2 + \sigma_0^2} \in [0, 1)",
            order=1
        ),
        Node(
            id="dual_limits",
            label="Asymptotic Gradient Bounds",
            container_id="homotopy_core",
            kind="pill_op",
            latex_math=r"s \to 0 \Rightarrow \|\nabla \mathcal{L}\| = \mathcal{O}(1)",
            order=2
        ),

        # Container 4: RoI Head & Loss
        Node(
            id="roi_align",
            label="RoIAlign Feature Pooling",
            container_id="roi_stage",
            kind="pill_op",
            latex_math=r"7 \times 7 \text{ Bilinear Sampling}",
            order=0
        ),
        Node(
            id="roi_mlp",
            label="Two-Layer MLP Head",
            container_id="roi_stage",
            kind="process",
            latex_math=r"2 \times 1024\text{-d FC}",
            order=1
        ),
        Node(
            id="box_loss",
            label="Bounded Homotopy Box Loss",
            container_id="roi_stage",
            kind="loss",
            latex_math=r"\mathcal{L}_{\text{H-WIoU}} = 1 - \mathcal{S}_{\text{H-WIoU}}(\hat{\mathbf{b}}, \mathbf{g})",
            order=2,
            emphasize=True
        ),
        Node(
            id="final_out",
            label="Final Detections",
            container_id="roi_stage",
            kind="io",
            latex_math=r"\mathcal{Y} = \{(\hat{y}_k, \hat{\mathbf{b}}_k, \hat{s}_k)\}",
            order=3
        ),
    ]

    edges = [
        # Dataflow Path 1: Feature Extraction
        Edge(source="input_img", target="resnet50", kind="flow"),
        Edge(source="resnet50", target="fpn", kind="flow"),

        # Dataflow Path 2: Stage 1 RPN & HLA
        Edge(source="fpn", target="anchor_gen", kind="flow", label="features"),
        Edge(source="anchor_gen", target="hla_module", kind="flow"),
        Edge(source="hla_module", target="rpn_proposals", kind="flow", label="positive matching"),

        # Homotopy Modulation Control Links
        Edge(source="homotopy_metric", target="scale_weight", kind="flow"),
        Edge(source="scale_weight", target="dual_limits", kind="flow"),
        Edge(source="homotopy_metric", target="hla_module", kind="reference", label="gamma(s) similarity"),
        Edge(source="homotopy_metric", target="box_loss", kind="reference", label="bounded supervision"),

        # Dataflow Path 3: Stage 2 RoI Head & Loss
        Edge(source="fpn", target="roi_align", kind="flow", label="P2-P5"),
        Edge(source="rpn_proposals", target="roi_align", kind="flow", label="RoIs"),
        Edge(source="roi_align", target="roi_mlp", kind="flow"),
        Edge(source="roi_mlp", target="box_loss", kind="flow", label="regression"),
        Edge(source="roi_mlp", target="final_out", kind="flow", label="class-aware NMS"),
    ]

    return UniversalDiagramSpec(
        title="Homotopy Wasserstein-IoU (H-WIoU) Detection Architecture",
        containers=containers,
        nodes=nodes,
        edges=edges
    )


def export_all_formats():
    """Export spec to Draw.io XML, TikZ, SVG, and clean publication renderings."""
    spec = build_rda_spec()
    style = UnifiedStyleSpec.load_preset("neurips_scientific_pastels")
    layout = compute_layout(spec)

    # 1. Draw.io XML
    drawio_path = FIG_DIR / "hwiou_pipeline_architecture.drawio"
    rda.render_drawio_xml(spec, style, layout, str(drawio_path))
    print(f"Exported Draw.io XML -> {drawio_path}")

    # 2. Standalone LaTeX TikZ
    tikz_path = FIG_DIR / "hwiou_pipeline_architecture.tex"
    rda.render_tikz(spec, style, layout, str(tikz_path))
    print(f"Exported LaTeX TikZ -> {tikz_path}")

    # 3. Responsive SVG
    svg_path = FIG_DIR / "hwiou_pipeline_architecture.svg"
    rda.render_svg(spec, style, layout, str(svg_path))
    print(f"Exported Vector SVG -> {svg_path}")


if __name__ == "__main__":
    export_all_formats()
