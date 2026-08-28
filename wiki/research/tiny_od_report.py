from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

W, H = A4

# ── Color palette ──────────────────────────────────────────────────────────────
NAVY    = colors.HexColor("#0D1B2A")
BLUE    = colors.HexColor("#1565C0")
ACCENT  = colors.HexColor("#1E88E5")
TEAL    = colors.HexColor("#00ACC1")
RED     = colors.HexColor("#E53935")
ORANGE  = colors.HexColor("#FB8C00")
GREEN   = colors.HexColor("#43A047")
GREY90  = colors.HexColor("#F5F7FA")
GREY70  = colors.HexColor("#ECEFF1")
GREY40  = colors.HexColor("#90A4AE")
WHITE   = colors.white
BLACK   = colors.black

# ── Page template with header/footer ──────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    # top bar
    canvas.setFillColor(NAVY)
    canvas.rect(0, H - 18*mm, W, 18*mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, H - 20*mm, W, 2*mm, fill=1, stroke=0)
    # header text
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(1.5*cm, H - 11*mm, "Tiny Object Detection — Architecture & Training Strategies")
    canvas.setFont("Helvetica", 7)
    canvas.drawRightString(W - 1.5*cm, H - 11*mm, "TinyPerson Detection Project · Faster R-CNN + RFLA")
    # footer
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, W, 10*mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 10*mm, W, 1*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(1.5*cm, 3.5*mm, "Deep Research Report · 2026-06-05")
    canvas.drawRightString(W - 1.5*cm, 3.5*mm, f"Page {doc.page}")
    canvas.restoreState()

def on_first_page(canvas, doc):
    canvas.saveState()
    # footer only on cover
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, W, 10*mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, 10*mm, W, 1*mm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(1.5*cm, 3.5*mm, "Deep Research Report · 2026-06-05")
    canvas.drawRightString(W - 1.5*cm, 3.5*mm, "TinyPerson Detection Project")
    canvas.restoreState()

# ── Styles ─────────────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()
    s = {}

    s['title_big'] = ParagraphStyle('title_big',
        fontName='Helvetica-Bold', fontSize=26, leading=32,
        textColor=WHITE, alignment=TA_LEFT, spaceAfter=6)

    s['title_sub'] = ParagraphStyle('title_sub',
        fontName='Helvetica', fontSize=13, leading=18,
        textColor=colors.HexColor("#90CAF9"), alignment=TA_LEFT, spaceAfter=4)

    s['tag'] = ParagraphStyle('tag',
        fontName='Helvetica-Bold', fontSize=8, leading=12,
        textColor=TEAL, alignment=TA_LEFT)

    s['h1'] = ParagraphStyle('h1',
        fontName='Helvetica-Bold', fontSize=15, leading=20,
        textColor=WHITE, alignment=TA_LEFT,
        spaceBefore=2, spaceAfter=6,
        leftIndent=0)

    s['h2'] = ParagraphStyle('h2',
        fontName='Helvetica-Bold', fontSize=11, leading=15,
        textColor=NAVY, alignment=TA_LEFT,
        spaceBefore=14, spaceAfter=4,
        borderPad=3)

    s['h3'] = ParagraphStyle('h3',
        fontName='Helvetica-Bold', fontSize=9.5, leading=14,
        textColor=BLUE, alignment=TA_LEFT,
        spaceBefore=10, spaceAfter=3)

    s['body'] = ParagraphStyle('body',
        fontName='Helvetica', fontSize=9, leading=14,
        textColor=colors.HexColor("#212121"), alignment=TA_JUSTIFY,
        spaceBefore=3, spaceAfter=4)

    s['body_bold'] = ParagraphStyle('body_bold',
        fontName='Helvetica-Bold', fontSize=9, leading=14,
        textColor=NAVY, alignment=TA_LEFT,
        spaceBefore=3, spaceAfter=2)

    s['bullet'] = ParagraphStyle('bullet',
        fontName='Helvetica', fontSize=9, leading=14,
        textColor=colors.HexColor("#212121"), alignment=TA_LEFT,
        spaceBefore=1, spaceAfter=1,
        leftIndent=14, firstLineIndent=-10)

    s['code'] = ParagraphStyle('code',
        fontName='Courier', fontSize=8, leading=12,
        textColor=colors.HexColor("#1A237E"),
        backColor=colors.HexColor("#E8EAF6"),
        alignment=TA_LEFT, spaceBefore=4, spaceAfter=4,
        leftIndent=8, rightIndent=8,
        borderPad=4)

    s['caption'] = ParagraphStyle('caption',
        fontName='Helvetica-Oblique', fontSize=7.5, leading=11,
        textColor=GREY40, alignment=TA_CENTER,
        spaceBefore=2, spaceAfter=6)

    s['callout'] = ParagraphStyle('callout',
        fontName='Helvetica-Bold', fontSize=9, leading=14,
        textColor=NAVY, alignment=TA_LEFT,
        leftIndent=12, rightIndent=8,
        spaceBefore=6, spaceAfter=6)

    s['verdict'] = ParagraphStyle('verdict',
        fontName='Helvetica-Bold', fontSize=9, leading=14,
        textColor=WHITE, alignment=TA_LEFT,
        leftIndent=6)

    s['toc_h'] = ParagraphStyle('toc_h',
        fontName='Helvetica-Bold', fontSize=10, leading=14,
        textColor=WHITE)

    s['toc_item'] = ParagraphStyle('toc_item',
        fontName='Helvetica', fontSize=9, leading=15,
        textColor=colors.HexColor("#CFD8DC"))

    s['metric'] = ParagraphStyle('metric',
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        textColor=TEAL, alignment=TA_CENTER)

    s['metric_label'] = ParagraphStyle('metric_label',
        fontName='Helvetica', fontSize=7.5, leading=11,
        textColor=GREY40, alignment=TA_CENTER)

    s['section_num'] = ParagraphStyle('section_num',
        fontName='Helvetica-Bold', fontSize=11, leading=14,
        textColor=TEAL, alignment=TA_LEFT)

    return s

# ── Custom Flowables ───────────────────────────────────────────────────────────
class SectionHeader(Flowable):
    """Colored band with section number + title."""
    def __init__(self, num, title, color=BLUE):
        super().__init__()
        self.num = num
        self.title = title
        self.color = color
        self.width = W - 3*cm
        self.height = 28

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        c.roundRect(0, 0, self.width, self.height, 4, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.roundRect(0, 0, 28, self.height, 4, fill=1, stroke=0)
        c.rect(24, 0, 4, self.height, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(14, 9, self.num)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(38, 9, self.title)

class CalloutBox(Flowable):
    """Highlighted insight box."""
    def __init__(self, text, icon="▶", color=TEAL, width=None):
        super().__init__()
        self.text = text
        self.icon = icon
        self.color = color
        self._width = width or (W - 3*cm)
        self.height = 36

    def wrap(self, availW, availH):
        self._width = availW
        return (availW, self.height)

    def draw(self):
        c = self.canv
        c.setFillColor(colors.HexColor("#E3F2FD"))
        c.roundRect(0, 0, self._width, self.height, 4, fill=1, stroke=0)
        c.setFillColor(self.color)
        c.rect(0, 0, 4, self.height, fill=1, stroke=0)
        c.setFillColor(self.color)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(12, self.height/2 - 5, self.icon)
        c.setFillColor(NAVY)
        c.setFont("Helvetica", 8.5)
        # wrap text manually
        words = self.text.split()
        line, lines = [], []
        for w in words:
            test = ' '.join(line + [w])
            if c.stringWidth(test, "Helvetica", 8.5) < self._width - 50:
                line.append(w)
            else:
                lines.append(' '.join(line))
                line = [w]
        lines.append(' '.join(line))
        y = self.height/2 + (len(lines)-1)*6 - 5
        for l in lines:
            c.drawString(28, y, l)
            y -= 12

class VerdictBox(Flowable):
    def __init__(self, priority, text, width=None):
        super().__init__()
        self.priority = priority
        self.text = text
        self._width = width or (W - 3*cm)
        self.height = 30
        self.color = {"HIGHEST": RED, "HIGH": ORANGE, "MEDIUM": ACCENT, "LOW": GREEN}.get(priority, GREY40)

    def wrap(self, availW, availH):
        self._width = availW
        return (availW, self.height)

    def draw(self):
        c = self.canv
        c.setFillColor(self.color)
        c.roundRect(0, 0, self._width, self.height, 4, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(8, self.height/2 - 4, f"PRIORITY: {self.priority}")
        c.setFont("Helvetica", 8)
        c.drawString(8, self.height/2 + 6, self.text)

# ── Table helpers ──────────────────────────────────────────────────────────────
def make_table(headers, rows, col_widths=None, zebra=True):
    data = [headers] + rows
    col_w = col_widths or ([( W - 3*cm) / len(headers)] * len(headers))
    t = Table(data, colWidths=col_w, repeatRows=1)
    style = [
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor("#CFD8DC")),
        ('ROWBACKGROUND', (0,0), (-1,0), NAVY),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]
    if zebra:
        for i in range(1, len(data)):
            bg = GREY90 if i % 2 == 0 else WHITE
            style.append(('BACKGROUND', (0,i), (-1,i), bg))
    t.setStyle(TableStyle(style))
    return t

def priority_table(headers, rows, col_widths=None):
    """Table with color-coded priority in last column."""
    data = [headers] + rows
    col_w = col_widths or ([(W - 3*cm) / len(headers)] * len(headers))
    t = Table(data, colWidths=col_w, repeatRows=1)
    prio_colors = {"🔴 HIGHEST": RED, "🔴 HIGH": ORANGE, "🟡 MEDIUM": ACCENT, "🟢 LOW": GREEN}
    style = [
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), WHITE),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7.5),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor("#CFD8DC")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]
    for i in range(1, len(data)):
        bg = GREY90 if i % 2 == 0 else WHITE
        style.append(('BACKGROUND', (0,i), (-1,i), bg))
    t.setStyle(TableStyle(style))
    return t

# ── Build ──────────────────────────────────────────────────────────────────────
OUTPUT = "wiki/research/tiny_od_architecture_report.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=1.5*cm, rightMargin=1.5*cm,
    topMargin=2.5*cm, bottomMargin=1.8*cm,
    title="Tiny Object Detection — Architecture & Training Strategies",
    author="TinyPerson Detection Project",
)

S = make_styles()
story = []

# ══════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ══════════════════════════════════════════════════════════════════════════════
def cover_page(canvas, doc):
    canvas.saveState()
    # dark background
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, W, H, fill=1, stroke=0)
    # accent stripe top
    canvas.setFillColor(TEAL)
    canvas.rect(0, H - 6*mm, W, 6*mm, fill=1, stroke=0)
    # hero gradient band
    canvas.setFillColor(colors.HexColor("#112240"))
    canvas.rect(0, H*0.35, W, H*0.48, fill=1, stroke=0)
    # left accent bar
    canvas.setFillColor(TEAL)
    canvas.rect(0, H*0.35, 6, H*0.48, fill=1, stroke=0)
    # bottom strip
    canvas.setFillColor(colors.HexColor("#0A1628"))
    canvas.rect(0, 0, W, H*0.35, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, H*0.35, W, 2, fill=1, stroke=0)
    # main title
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 30)
    canvas.drawString(1.8*cm, H*0.72, "Tiny Object Detection")
    canvas.setFont("Helvetica-Bold", 22)
    canvas.setFillColor(colors.HexColor("#90CAF9"))
    canvas.drawString(1.8*cm, H*0.65, "Architecture & Training Strategies")
    # subtitle
    canvas.setFillColor(TEAL)
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(1.8*cm, H*0.60, "Deep Research Report")
    # decorative line
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(2)
    canvas.line(1.8*cm, H*0.58, 12*cm, H*0.58)
    # dataset stats row
    stats = [
        ("92%", "Tiny/Small\nObjects (<32px)"),
        ("27%", "Micro Objects\n(<8px)"),
        ("45", "Avg Objects\nper Image"),
        ("0.0428", "Current\nAP@75"),
    ]
    x0 = 1.8*cm
    bw = (W - 3.6*cm) / 4
    for i, (val, lbl) in enumerate(stats):
        x = x0 + i * bw
        canvas.setFillColor(colors.HexColor("#1E2D4A"))
        canvas.roundRect(x, H*0.47, bw - 6, H*0.09, 4, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawCentredString(x + (bw-6)/2, H*0.505, val)
        canvas.setFillColor(GREY40)
        canvas.setFont("Helvetica", 7)
        for j, line in enumerate(lbl.split('\n')):
            canvas.drawCentredString(x + (bw-6)/2, H*0.478 - j*9, line)
    # project info
    canvas.setFillColor(colors.HexColor("#546E7A"))
    canvas.setFont("Helvetica", 8.5)
    canvas.drawString(1.8*cm, H*0.43, "Faster R-CNN  ·  RFLA Label Assignment  ·  SAH-GD Metric  ·  P2 FPN Level")
    canvas.setFont("Helvetica-Bold", 8.5)
    canvas.setFillColor(colors.HexColor("#78909C"))
    canvas.drawString(1.8*cm, H*0.39, "Current best: HARD_SWITCH  mAP(scale)=0.5770  |  AP_micro=0.2776  |  AP@75=0.0428 ← bottleneck")
    # tags
    tags = ["#RoIAlign", "#CascadeRCNN", "#DIoU", "#MultiScale", "#LabelAssignment", "#TinyObjects"]
    tx = 1.8*cm
    ty = H*0.33
    for tag in tags:
        tw = canvas.stringWidth(tag, "Helvetica-Bold", 7.5) + 10
        canvas.setFillColor(colors.HexColor("#1A3050"))
        canvas.roundRect(tx, ty, tw, 14, 3, fill=1, stroke=0)
        canvas.setFillColor(TEAL)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(tx + 5, ty + 3.5, tag)
        tx += tw + 6
    # footer
    canvas.setFillColor(colors.HexColor("#546E7A"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(1.8*cm, 1.5*cm, "Generated: 2026-06-05  ·  For internal use — TinyPerson Detection Project")
    canvas.restoreState()

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# TABLE OF CONTENTS
# ══════════════════════════════════════════════════════════════════════════════
toc_data = [
    ("01", "High-Resolution RoIAlign for Tiny Objects",        "Why 7×7 fails, what the literature says, recommended config"),
    ("02", "Cascade R-CNN for Small/Tiny Objects",             "Iterative refinement, IoU threshold tuning, 2-stage vs 3-stage"),
    ("03", "Regression Parametrization Alternatives",          "Gaussian similarity, DIoU/CIoU, dual-objective scheduling"),
    ("04", "Multi-Scale Training & Inference Strategies",      "SNIP/SNIPER, mosaic, copy-paste, TTA"),
    ("05", "Label Assignment for Tiny Objects",                "ATSS, SimOTA, PAA, scale-adaptive dynamic k"),
    ("06", "Synthesis & Priority Recommendations",             "Priority matrix, execution order, key principles"),
]

toc_table_data = []
for num, title, desc in toc_data:
    toc_table_data.append([
        Paragraph(f'<font color="#00ACC1"><b>{num}</b></font>', S['body']),
        Paragraph(f'<b>{title}</b><br/><font color="#90A4AE" size="7">{desc}</font>', S['body']),
    ])

toc_t = Table(toc_table_data, colWidths=[1.2*cm, W - 4.7*cm])
toc_t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#112240")),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor("#1E3A5F")),
    ('ROUNDEDCORNERS', [4, 4, 4, 4]),
    ('TEXTCOLOR', (0,0), (-1,-1), WHITE),
]))

# ToC page header (manual)
story.append(Spacer(1, 0.3*cm))

toc_header = Table(
    [[Paragraph("Contents", S['h1'])]],
    colWidths=[W - 3*cm]
)
toc_header.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), NAVY),
    ('TOPPADDING', (0,0), (-1,-1), 8),
    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ('LEFTPADDING', (0,0), (-1,-1), 12),
    ('ROUNDEDCORNERS', [4, 4, 4, 4]),
]))
story.append(toc_header)
story.append(Spacer(1, 0.4*cm))
story.append(toc_t)
story.append(Spacer(1, 0.6*cm))

# Context box
ctx_data = [[
    Paragraph("<b>Project Context</b><br/>"
              "Gaussian regression loss <font face='Courier' size='8'>1−exp(−β·D_H)</font> is IoU-insensitive by design. "
              "This is the structural root cause of the stuck <b>AP@75 = 0.0428</b>. "
              "Dataset: 92% tiny/small, 27% micro (&lt;8px), avg 45 obj/img. "
              "All recommendations in this report are anchored to this specific constraint.", S['body'])
]]
ctx_t = Table(ctx_data, colWidths=[W - 3*cm])
ctx_t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFF3E0")),
    ('LEFTPADDING', (0,0), (-1,-1), 10),
    ('TOPPADDING', (0,0), (-1,-1), 8),
    ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ('LINECOLOR', (0,0), (0,-1), ORANGE),
    ('LINEBEFORE', (0,0), (0,-1), 4, ORANGE),
]))
story.append(ctx_t)
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RoIAlign
# ══════════════════════════════════════════════════════════════════════════════
story.append(SectionHeader("01", "High-Resolution RoIAlign for Tiny Objects", NAVY))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("1.1  Why Standard 7×7 Fails for Micro Objects", S['h2']))
story.append(Paragraph(
    "Standard Faster R-CNN uses RoIAlign(output_size=7, sampling_ratio=2). "
    "For tiny objects, the number of feature cells covered by a proposal is far too small to carry useful spatial information.",
    S['body']))

roi_rows = [
    ["32 px", "P3", "8", "4×4", "Moderate — upsampled from 4→7"],
    ["16 px", "P3", "8", "2×2", "Poor — severe upsampling"],
    ["8 px",  "P3", "8", "1×1", "Useless — single cell → constant 7×7"],
    ["8 px",  "P2", "4", "2×2", "Still very coarse"],
    ["6 px",  "P2", "4", "~1.5×1.5", "Sub-pixel — only ~4 cells contribute"],
]
story.append(make_table(
    ["Object Size", "FPN Level", "Stride", "Feature Cells Covered", "7×7 Pool Quality"],
    roi_rows,
    col_widths=[2.2*cm, 2*cm, 1.5*cm, 3.5*cm, 7.3*cm]
))
story.append(Paragraph(
    "Key insight: Even at P2 (stride-4), a 6px object covers only ~1.5 feature cells. "
    "The 7×7 RoIAlign output is a bilinearly interpolated blow-up with almost no spatial variation — "
    "the box head has nothing useful to regress from. This directly explains the stuck AP@75.",
    S['body']))

story.append(Paragraph("1.2  Literature Evidence for Higher-Resolution RoIAlign", S['h2']))
lit_rows = [
    ["Mask R-CNN (2017)", "14×14 RoIAlign", "for mask head (keeps 7×7 for box head)", "Baseline precedent"],
    ["HRDNet (2021)", "14×14 / 28×28", "+1.5–2.0 AP on COCO small (14×14)", "Direct evidence"],
    ["Grid R-CNN (2019)", "14×14 / 28×28 grid", "+1.3 AP on COCO via grid-point prediction", "Spatial exploitation"],
    ["TridentNet (2019)", "Scale-specific branches", "Dilation-rate variant of the same principle", "Scale-aware design"],
]
story.append(make_table(
    ["Paper", "RoIAlign Config", "Finding", "Relevance"],
    lit_rows, col_widths=[3.2*cm, 3*cm, 6.5*cm, 3.8*cm]
))

story.append(Paragraph("1.3  Per-Level Adaptive Strategy", S['h2']))
story.append(Paragraph(
    "The optimal output_size should vary by FPN level. The recommended approach uses 14×14 globally "
    "with a lightweight conv head (1–2 conv layers with stride 2) before the FC layers:",
    S['body']))
lvl_rows = [
    ["P2", "4", "4–12 px", "1–3 cells", "14×14 or 21×21 — CRITICAL"],
    ["P3", "8", "8–24 px", "1–3 cells", "14×14 — HIGH BENEFIT"],
    ["P4", "16", "24–48 px", "1.5–3 cells", "7×7 — sufficient"],
    ["P5", "32", "48–96 px", "1.5–3 cells", "7×7 — sufficient"],
]
story.append(make_table(
    ["FPN Level", "Stride", "Typical Object", "Feature Cells", "Recommended output_size"],
    lvl_rows, col_widths=[2*cm, 1.5*cm, 2.5*cm, 2.5*cm, 7.5*cm]
))

story.append(Paragraph("1.4  Quantitative Impact Expectations", S['h2']))
impact_rows = [
    ["7×7 → 14×14, plain FC head", "+0.5–1.5 AP@75", "More spatial info; FC may not exploit it"],
    ["7×7 → 14×14, conv head (2–3 conv + pool)", "+1.5–3.0 AP@75", "Conv layers extract spatial patterns before regression"],
    ["14×14 + sampling_ratio 2→4", "+0.3–0.5 AP@75", "Better sub-pixel interpolation quality"],
    ["Per-level adaptive (14 for P2–P3, 7 for P4–P5)", "Similar to global 14", "Lower compute, avoids waste on easy levels"],
]
story.append(make_table(
    ["Configuration", "Expected AP@75 Δ", "Mechanism"],
    impact_rows, col_widths=[6*cm, 3.2*cm, 7.3*cm]
))

warn_data = [[Paragraph(
    "<b>⚠ Critical Caveat:</b> Higher RoIAlign alone is necessary but NOT sufficient. "
    "If the loss function does not reward precise localization, the head will not learn to use "
    "the additional spatial information. RoIAlign 14×14 MUST be combined with an IoU-aware "
    "regression loss (Section 03) to realize the AP@75 gain.", S['body'])]]
warn_t = Table(warn_data, colWidths=[W - 3*cm])
warn_t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFF8E1")),
    ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ('LEFTPADDING', (0,0), (-1,-1), 10),
    ('LINEBEFORE', (0,0), (0,-1), 4, ORANGE),
]))
story.append(Spacer(1, 0.2*cm))
story.append(warn_t)

story.append(Paragraph("1.5  Verdict & Recommended Configuration", S['h2']))
story.append(Paragraph(
    "<b>High Priority.</b> The 7×7 RoIAlign is a clear structural bottleneck for objects &lt;12px. "
    "Recommended implementation (Notebook 14 pattern):", S['body']))
cfg_rows = [
    ["RoIAlign output_size", "14", "sampling_ratio", "4"],
    ["Conv head", "Conv3×3-256-ReLU → Conv3×3-256-ReLU → AdaptiveAvgPool(7)", "then", "2×FC-1024"],
    ["Must combine with", "DIoU/CIoU regression term (Section 03)", "Memory overhead", "~4× per RoI (manageable)"],
]
story.append(make_table(["Parameter", "Value", "Parameter", "Value"], cfg_rows,
    col_widths=[3.5*cm, 7.5*cm, 3*cm, 2.5*cm]))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Cascade R-CNN
# ══════════════════════════════════════════════════════════════════════════════
story.append(SectionHeader("02", "Cascade R-CNN for Small/Tiny Objects", NAVY))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("2.1  Core Mechanism", S['h2']))
story.append(Paragraph(
    "Cascade R-CNN (Cai & Vasconcelos, CVPR 2018) addresses the mismatch between training IoU thresholds "
    "and test-time quality. Each stage uses a progressively stricter IoU threshold, with proposals from "
    "the previous stage as input:",
    S['body']))
flow_data = [
    [Paragraph("<b>Standard</b>", S['body']),
     Paragraph("RPN → RoI Head (IoU=0.5) → output", S['code'])],
    [Paragraph("<b>2-stage Cascade</b>", S['body']),
     Paragraph("RPN → Stage 1 (IoU=0.5) → Stage 2 (IoU=0.6) → output", S['code'])],
    [Paragraph("<b>3-stage Cascade</b>", S['body']),
     Paragraph("RPN → Stage 1 (IoU=0.5) → Stage 2 (IoU=0.6) → Stage 3 (IoU=0.7) → output", S['code'])],
]
flow_t = Table(flow_data, colWidths=[3.5*cm, W - 6.5*cm])
flow_t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,-1), GREY90),
    ('BACKGROUND', (1,0), (1,-1), colors.HexColor("#E8EAF6")),
    ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor("#CFD8DC")),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(flow_t)

story.append(Paragraph("2.2  The IoU Problem for Tiny Boxes", S['h2']))
story.append(Paragraph(
    "For tiny objects, IoU is extremely sensitive to pixel-level shifts. This fundamentally challenges "
    "the cascade assumption that successive stages can achieve higher IoU:", S['body']))
iou_rows = [
    ["32 px", "1 px", "0.88", "Achievable with stride-8 features"],
    ["16 px", "1 px", "0.77", "Needs precision"],
    ["16 px", "2 px", "0.59", "1-stage threshold already hard"],
    ["8 px",  "1 px", "0.59", "Sub-pixel precision required"],
    ["8 px",  "2 px", "0.36", "Falls below even Stage 1 threshold"],
    ["6 px",  "1 px", "0.47", "Already below IoU=0.5"],
    ["6 px",  "2 px", "0.24", "Stage 3 at IoU=0.7 is impossible"],
]
story.append(make_table(
    ["Object Size", "Shift δ", "Resulting IoU", "Interpretation"],
    iou_rows, col_widths=[2.5*cm, 2*cm, 2.5*cm, 9.5*cm]
))

story.append(Paragraph("2.3  Threshold Configurations", S['h2']))
cas_rows = [
    ["2-stage standard", "[0.5, 0.6]", "Conservative; positives at 0.6 for tiny", "Still too tight for <8px"],
    ["2-stage tiny-tuned", "[0.4, 0.55]", "More tiny positives at both stages", "Recommended for this project"],
    ["3-stage standard", "[0.5, 0.6, 0.7]", "Maximum refinement for large objects", "Stage 3 starved of tiny positives"],
    ["3-stage tiny-tuned", "[0.4, 0.5, 0.6]", "Reasonable tiny coverage", "IoU=0.6 still challenging for <8px"],
]
story.append(make_table(
    ["Config", "IoU Thresholds", "Pros", "Cons"],
    cas_rows, col_widths=[3.5*cm, 2.8*cm, 5.5*cm, 4.7*cm]
))

story.append(Paragraph("2.4  Efficiency Considerations (Kaggle T4 constraint)", S['h2']))
eff_rows = [
    ["Parameters",   "+50% RoI head", "+100% RoI head"],
    ["Training time","+30–40%",        "+50–70%"],
    ["Inference time","+20–30%",       "+40–50%"],
    ["Memory",       "+25–35%",        "+45–60%"],
]
story.append(make_table(["Component", "2-Stage Overhead", "3-Stage Overhead"],
    eff_rows, col_widths=[4*cm, 5*cm, 5*cm]))

story.append(Paragraph("2.5  Verdict", S['h2']))
story.append(Paragraph(
    "<b>Medium priority — try after RoIAlign fix and regression loss fix.</b> "
    "Cascade adds value but requires: (1) lowering IoU thresholds to [0.4, 0.55] for tiny scale, "
    "(2) adapting to Gaussian-distance matching if possible, (3) 2-stage preferred over 3-stage "
    "for memory and positive-sample reasons.", S['body']))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Regression
# ══════════════════════════════════════════════════════════════════════════════
story.append(SectionHeader("03", "Regression Parametrization — THE AP@75 Lever", RED))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("3.1  Root Cause Analysis", S['h2']))
story.append(Paragraph(
    "The project uses Gaussian similarity regression: <font face='Courier'>L_reg = 1 − exp(−β·D_H)</font>. "
    "This function has a fundamental structural property that prevents AP@75 from improving:", S['body']))
prop_rows = [
    ["✅ Scale-invariant", "Good — no explosion for tiny boxes"],
    ["✅ Stable gradients", "Good — no scale-dependent loss magnitude"],
    ["❌ IoU-insensitive", "CRITICAL — a 1–2px error on a 6px box gives high similarity; loss is 'satisfied'"],
    ["❌ Vanishing gradient at extremes", "Hard cases (highly misaligned proposals) get weak gradient signal"],
]
prop_t = Table(prop_rows, colWidths=[5*cm, W - 8*cm])
prop_t.setStyle(TableStyle([
    ('FONTNAME', (0,0), (-1,-1), 'Helvetica'), ('FONTSIZE', (0,0), (-1,-1), 8.5),
    ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('BACKGROUND', (0,0), (0,1), colors.HexColor("#E8F5E9")),
    ('BACKGROUND', (0,2), (0,3), colors.HexColor("#FFEBEE")),
    ('BACKGROUND', (1,0), (1,1), colors.HexColor("#F1F8E9")),
    ('BACKGROUND', (1,2), (1,3), colors.HexColor("#FFF3F3")),
    ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor("#CFD8DC")),
]))
story.append(prop_t)

story.append(Paragraph("3.2  Comparative Loss Analysis for Tiny Box Regression", S['h2']))
loss_rows = [
    ["Smooth-L1 (standard)", "No", "N/A", "Moderate", "Poor"],
    ["Gaussian similarity", "Yes", "Yes (Wasserstein)", "NO — saturates", "Excellent"],
    ["GIoU", "Yes", "Weak (slow)", "Good", "Moderate"],
    ["DIoU", "Yes", "STRONG (center)", "Good", "Good ✓"],
    ["CIoU", "Yes", "Strong", "Best (aspect ratio)", "Moderate"],
    ["NWD/GCD raw", "Yes", "Yes", "No (same as Gaussian)", "Excellent"],
]
story.append(make_table(
    ["Loss", "Scale Invariant?", "Signal at Zero Overlap?", "Precise at High IoU?", "Stability <8px?"],
    loss_rows, col_widths=[3.5*cm, 2.5*cm, 3.5*cm, 4*cm, 3*cm]
))

story.append(Paragraph("3.3  The Dual-Objective Solution", S['h2']))
story.append(Paragraph(
    "The correct fix is to keep the Gaussian similarity term (for stability and scale-invariance) "
    "and add a <b>precision term</b> (IoU-aware) that activates at high overlap:", S['body']))
code_text = (
    "L_reg = (1 − S_H)          # Gaussian similarity: coarse, stable, scale-invariant\n"
    "      + γ · L_precision     # IoU-aware precision term\n\n"
    "# Recommended: DIoU loss\n"
    "# DIoU = IoU − ρ²(b_pred, b_gt) / c²\n"
    "# Center distance penalty provides gradient even at zero overlap\n"
    "# Critical for 6–8px objects where initial proposals often have zero IoU"
)
story.append(Paragraph(code_text.replace('\n', '<br/>'), S['code']))

story.append(Paragraph("3.4  Why DIoU Over CIoU for This Project", S['h2']))
story.append(Paragraph(
    "For 6–8px objects, center penalty is the dominant need. A 2px center error → IoU of 0.24–0.36. "
    "CIoU adds an aspect-ratio term that is <b>quantization-sensitive for tiny boxes</b>: "
    "the difference between arctan(6/10) and arctan(7/10) is only 0.007 — dominated by pixel noise. "
    "DIoU's center penalty provides non-zero gradient for non-overlapping proposals, which is critical "
    "when initial RPN proposals are often completely misaligned with micro-object GTs.", S['body']))

story.append(Paragraph("3.5  Scheduling Strategy (Critical Implementation Detail)", S['h2']))
story.append(Paragraph(
    "Notebook 10 failure (GAMMA_FINE=1.0 from epoch 1) demonstrates that applying the IoU-aware term "
    "too early or too strongly causes training instability. Early proposals are so imprecise that "
    "IoU-based gradients are noisy. The recommended schedule:", S['body']))
sched_code = (
    "if epoch < warmup_epochs (=3):\n"
    "    L = 1 − S_H                              # Gaussian only\n"
    "elif epoch < transition_end (=8):\n"
    "    γ_eff = γ · (epoch − 3) / (8 − 3)        # linear ramp\n"
    "    L = (1 − S_H) + γ_eff · DIoU_loss\n"
    "else:\n"
    "    L = (1 − S_H) + γ · DIoU_loss            # full dual (γ = 0.3–0.5)"
)
story.append(Paragraph(sched_code.replace('\n', '<br/>'), S['code']))

story.append(Paragraph("3.6  Verdict", S['h2']))
story.append(Paragraph(
    "<b>HIGHEST PRIORITY — this is THE lever for AP@75.</b> "
    "No architectural change can fix AP@75 if the loss doesn't reward precision. "
    "Recommended: DIoU as the precision term, γ=0.3–0.5, scheduled ramp-up starting epoch 3–4. "
    "Keep Gaussian similarity as the primary term for stability.", S['body']))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Multi-Scale
# ══════════════════════════════════════════════════════════════════════════════
story.append(SectionHeader("04", "Multi-Scale Training & Inference Strategies", NAVY))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("4.1  The Scale Problem", S['h2']))
story.append(Paragraph(
    "The TinyPerson dataset spans a 144× scale range (2px–289px). "
    "27% of objects are micro (&lt;8px), 52% are micro+tiny (&lt;12px). "
    "Standard single-resolution training forces the network to handle all scales simultaneously — "
    "a structural disadvantage for micro-object recognition.", S['body']))

story.append(Paragraph("4.2  SNIP / SNIPER", S['h2']))
snip_rows = [
    ["SNIP (CVPR 2018)", "Image pyramid — train only on objects in the 'valid range' for each scale",
     "Sound philosophy; prohibitive memory for TinyPerson (4–8× upscale needed for micro objects)"],
    ["SNIPER (NeurIPS 2018)", "Chip-based SNIP — crop fixed-size chips around object clusters at each scale",
     "Memory-feasible; significant implementation complexity for RFLA + Faster R-CNN setup"],
]
story.append(make_table(["Method", "Core Idea", "Assessment for This Project"],
    snip_rows, col_widths=[3*cm, 7*cm, 6.5*cm]))

story.append(Paragraph("4.3  Practical Multi-Scale Approaches", S['h2']))
ms_rows = [
    ["Multi-resolution training\n(random scale)", "Each iteration: random resolution from {512, 640, 768, 896, 1024}",
     "Easy — MIN_SIZE/MAX_SIZE already supported", "Moderate", "✅ DO THIS"],
    ["Copy-paste augmentation", "Extract micro crops, paste at random positions in other images",
     "Maintains object size; more controlled than mosaic", "Medium", "✅ Worth trying"],
    ["Mosaic (crop, not resize)", "Combine 4 image crops without downscaling",
     "Maintains pixel size; avoid resize variant which shrinks objects further", "Medium", "⚠ With care"],
    ["Multi-scale TTA (inference)", "Run at 1.0×, 1.5×, 2.0× and merge with NMS",
     "Too slow for Kaggle time limit; useful for offline eval", "High", "🔬 Offline only"],
]
story.append(make_table(
    ["Strategy", "Mechanism", "Assessment", "Complexity", "Recommendation"],
    ms_rows, col_widths=[3.2*cm, 5*cm, 4.5*cm, 2.3*cm, 1.5*cm]  # fixed
))
story.append(Paragraph(
    "⚠ Mosaic warning: Standard mosaic resizes each quadrant, which for 1920×1080 images at 1024px target "
    "shrinks objects by ~2× (4px objects → 2px — undetectable). Always use the crop variant (512×512 patches).",
    S['body']))

story.append(Paragraph("4.4  Verdict", S['h2']))
story.append(Paragraph(
    "<b>Medium priority, easy wins available.</b> "
    "Multi-resolution training is the default that should have been enabled from the start — "
    "minimal code change, consistent small improvement. Copy-paste augmentation is the right choice "
    "for micro objects specifically. SNIP/SNIPER provide the largest theoretical benefit but require "
    "significant engineering effort — defer for now.", S['body']))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Label Assignment
# ══════════════════════════════════════════════════════════════════════════════
story.append(SectionHeader("05", "Label Assignment for Tiny Objects", NAVY))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("5.1  Why Assignment Matters Disproportionately", S['h2']))
story.append(Paragraph(
    "For tiny objects, label assignment is critical for three compounding reasons:", S['body']))
why_rows = [
    ["1", "Few anchors overlap",
     "An 8px GT on P3 (stride-8) has IoU>0.5 with at most 1–2 anchors. "
     "Many GTs get ZERO positives under IoU-based assignment → undetectable."],
    ["2", "Extreme positive-negative ratio",
     "160×90 feature map at P3 → 14,400 locations. 45 GTs/image → <0.3% positive ratio. "
     "Classifier learns 'everything is background'."],
    ["3", "IoU is fundamentally unreliable",
     "For a 6×6 box, 1px shift changes IoU from 0.69 to 0.56 — massive noise. "
     "IoU threshold assignment is dominated by quantization noise for <8px objects."],
]
story.append(make_table(["#", "Problem", "Impact"], why_rows,
    col_widths=[0.7*cm, 4*cm, 11.8*cm]))

story.append(Paragraph("5.2  Method Comparison for &lt;8px Objects", S['h2']))
meth_rows = [
    ["Fixed IoU (0.5)",       "0–1",  "IoU",                    "YES, often", "Full",   "Low"],
    ["ATSS (k=9)",            "3–5",  "IoU (adaptive thresh.)", "Rare",       "High",   "Low"],
    ["SimOTA",                "0–3",  "Cost (cls+reg)",         "YES if IoU≈0","High",  "Medium"],
    ["PAA",                   "2–4",  "GMM on IoU×cls",         "Rare/noisy", "High",   "High"],
    ["Dynamic k (scale-adap.)","6–9", "RF distance",            "NO ✓",       "None",   "Low"],
    ["RFLA (this project)",   "3–9",  "Gaussian RF dist.",      "NO ✓",       "None ✓", "Low"],
]
story.append(make_table(
    ["Method", "Typical k (6px GT)", "Quality Metric", "Assigns 0 Positives?", "IoU Dependence", "Compute"],
    meth_rows, col_widths=[3*cm, 2.8*cm, 3.5*cm, 3.2*cm, 2.8*cm, 1.2*cm]
))

story.append(Paragraph("5.3  What Actually Works for &lt;8px Objects", S['h2']))
wins = [
    ("Replace IoU with scale-invariant metric (NWD/GCD) in matching.",
     "Single most impactful change. IoU is fundamentally broken for <8px — "
     "quantization noise exceeds signal. ✅ Already done in this project via RFLA + NWD/GCD."),
    ("Guarantee minimum positives per GT (k≥1).",
     "SimOTA and PAA can fail to assign any positives for micro objects when all IoU < 0.1. "
     "Dynamic k with a floor is essential."),
    ("Scale-adaptive k (more positives for smaller objects).",
     "Validated by project experiments. Micro objects need more positives: coarser features "
     "and noisier loss landscape. SCALE_TOPK won AP_micro=0.2947 in SAH-GD ablation."),
    ("Anchor/feature resolution > assignment sophistication.",
     "P2 result (+29% AP_micro) shows that giving micro objects actual feature resolution "
     "(stride-4 vs stride-8) has larger impact than any assignment change."),
]
for title_w, body_w in wins:
    story.append(Paragraph(f"<b>▶ {title_w}</b>", S['body_bold']))
    story.append(Paragraph(body_w, S['bullet']))

story.append(Paragraph("5.4  Recommended Configuration", S['h2']))
cfg2_code = (
    "# Label assignment: RFLA with scale-adaptive k\n"
    "ASSIGNMENT_METRIC = 'gaussian_rf_distance'  # NOT IoU\n"
    "K_MICRO  = 6    # objects < 8px   (was 9; reduced to avoid noisy positives with P2)\n"
    "K_TINY   = 5    # objects 8–20px\n"
    "K_OTHER  = 3    # objects > 20px\n"
    "MIN_K    = 1    # guarantee at least 1 positive per GT"
)
story.append(Paragraph(cfg2_code.replace('\n', '<br/>'), S['code']))

story.append(Paragraph("5.5  Verdict", S['h2']))
story.append(Paragraph(
    "<b>Already well-addressed.</b> RFLA + scale-adaptive k + Gaussian distance assignment is "
    "near-optimal for this dataset. Remaining gains are in: (1) tuning k values (k_micro=6 > k_micro=9 "
    "per notebook 12), (2) co-optimizing with P2 feature resolution, "
    "(3) potentially adding quality-weighted selection within top-k.", S['body']))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Synthesis & Recommendations
# ══════════════════════════════════════════════════════════════════════════════
story.append(SectionHeader("06", "Synthesis & Priority Recommendations", TEAL))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("6.1  Priority Matrix", S['h2']))
prio_rows = [
    ["DIoU regression (scheduled γ=0.3–0.5)", "+2–4%", "±0%", "Low",       "🔴 HIGHEST"],
    ["RoIAlign 14×14 + conv head",            "+1–2%", "+1–3%", "Medium",  "🔴 HIGH"],
    ["DIoU + RoIAlign 14×14 combined",        "+3–5%", "+1–3%", "Medium",  "🔴 HIGH"],
    ["Multi-resolution training",             "+0.5–1%", "+1–2%", "Low",   "🟡 MEDIUM"],
    ["2-stage Cascade (tuned thresholds)",    "+1–3%", "+0–1%", "High",    "🟡 MEDIUM"],
    ["Copy-paste augmentation",              "+0–1%", "+2–4%", "Medium",   "🟡 MEDIUM"],
    ["Scale-adaptive k tuning",             "+0–0.5%", "+1–2%", "Low",    "🟢 LOW (already good)"],
    ["3-stage Cascade",                     "+1–2%", "−1–0%", "Very High", "🟢 LOW"],
    ["SNIP/SNIPER",                         "+0.5–1%", "+2–3%", "Very High","🟢 LOW (defer)"],
    ["Mosaic augmentation",                 "+0–0.5%", "+1–2%", "Medium",  "🟢 LOW"],
]
story.append(make_table(
    ["Intervention", "Expected AP@75 Δ", "Expected AP_micro Δ", "Complexity", "Priority"],
    prio_rows, col_widths=[6.5*cm, 2.8*cm, 3.2*cm, 2.5*cm, 3.5*cm]
))

story.append(Paragraph("6.2  Recommended Execution Order", S['h2']))
phase_rows = [
    ["Phase 1\n(Fix AP@75 bottleneck)",
     "1. (a) DIoU regression with scheduled ramp-up (γ=0.3, warmup=3 epochs) — on clean P2F baseline\n"
     "2. (b) RoIAlign 14×14 + conv head — on clean P2F baseline\n"
     "3. (c) Combine winner of (a)+(b) → Expected: AP@75 > 0.06, mAP(scale) ≥ 0.58",
     "1–2 experiments"],
    ["Phase 2\n(Stack micro gains)",
     "4. (d) Scale-adaptive k tuning (k_micro=6) — on Phase 1 winner\n"
     "5. (e) Multi-resolution training (random 640–1024) — on Phase 1 winner",
     "2–3 experiments"],
    ["Phase 3\n(Advanced, only if plateaued)",
     "6. (f) 2-stage Cascade with Gaussian-distance thresholds\n"
     "7. (g) Copy-paste augmentation for micro objects\n"
     "8. (h) CIoU instead of DIoU (if aspect ratio precision needed)",
     "3–4 experiments"],
]
phase_t = Table(phase_rows, colWidths=[3.5*cm, 10.5*cm, 2.5*cm])
phase_t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,0), RED),
    ('BACKGROUND', (0,1), (0,1), ORANGE),
    ('BACKGROUND', (0,2), (0,2), GREY40),
    ('TEXTCOLOR', (0,0), (0,-1), WHITE),
    ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
    ('FONTNAME', (1,0), (-1,-1), 'Helvetica'),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ('LEFTPADDING', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CFD8DC")),
]))
story.append(phase_t)

story.append(Paragraph("6.3  Key Principles", S['h2']))
principles = [
    ("Fix the loss function first.",
     "No architectural change can fix AP@75 if the loss doesn't reward precision. "
     "The Gaussian similarity loss is the root cause — adding a DIoU term is the minimum necessary fix."),
    ("One variable at a time.",
     "Notebook 9's P2 result was confounded by 5 simultaneous changes. "
     "Each intervention should be tested in isolation on the P2F baseline."),
    ("RoIAlign resolution and regression loss are complementary, not alternatives.",
     "Higher RoIAlign gives the head more spatial information to work with; "
     "IoU-aware loss teaches it to use that information for precise boxes. Neither alone is sufficient."),
    ("Label assignment is already near-optimal.",
     "RFLA + Gaussian distance + scale-adaptive k is well-suited to this dataset. "
     "Further assignment changes have diminishing returns — the bottleneck is now in the head."),
    ("Multi-scale training is easy wins.",
     "Random resolution training requires minimal code changes and provides consistent small improvements. "
     "It should be the default, not an experiment."),
]
for i, (title_p, body_p) in enumerate(principles, 1):
    story.append(Paragraph(f"<b>{i}. {title_p}</b>", S['body_bold']))
    story.append(Paragraph(body_p, S['bullet']))

story.append(Spacer(1, 0.5*cm))
# References
story.append(HRFlowable(width="100%", thickness=1, color=GREY40))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph("References", S['h2']))
refs = [
    "He, K. et al. Mask R-CNN. ICCV 2017.",
    "Cai, Z. & Vasconcelos, N. Cascade R-CNN. CVPR 2018.",
    "Singh, B. & Davis, L.S. SNIP: Scale Normalization for Image Pyramids. CVPR 2018.",
    "Singh, B. et al. SNIPER: Efficient Multi-Scale Training. NeurIPS 2018.",
    "Rezatofighi, H. et al. Generalized Intersection over Union. CVPR 2019.",
    "Lu, X. et al. Grid R-CNN. CVPR 2019.",
    "Li, Y. et al. TridentNet: Scale-Aware Trident Networks. ICCV 2019.",
    "Zhang, S. et al. ATSS: Bridging Anchor-based and Anchor-free Detection. CVPR 2020.",
    "Kim, K. & Lee, H.S. PAA: Probabilistic Anchor Assignment. ECCV 2020.",
    "Zheng, Z. et al. DIoU/CIoU: Distance-IoU Loss. AAAI 2020.",
    "Bochkovskiy, A. et al. YOLOv4. 2020.",
    "Ge, Z. et al. YOLOX. 2021.",
    "Ghiasi, G. et al. Simple Copy-Paste Augmentation. CVPR 2021.",
    "Wang, J. et al. NWD: Normalized Wasserstein Distance for Tiny Object Detection. 2021.",
    "Xu, C. et al. RFLA: Gaussian Receptive Field based Label Assignment. ECCV 2022.",
    "Yang, Z. et al. GCD: Gaussian Combined Distance for Tiny Object Detection. 2023.",
]
ref_text = "  ·  ".join([f"[{i+1}] {r}" for i, r in enumerate(refs)])
story.append(Paragraph(ref_text, ParagraphStyle('refs',
    fontName='Helvetica', fontSize=7, leading=11,
    textColor=GREY40, alignment=TA_LEFT)))

# ── Build with cover ───────────────────────────────────────────────────────────
doc.build(story, onFirstPage=cover_page, onLaterPages=on_page)
print("PDF created:", OUTPUT)
