"""Draw the paper-ready GDN architecture figure as a vector PDF.

The diagram is tied to ``recipe_model.py`` and the pinned Zoology-style
backbone in ``third_party/zoology_iclr24/model.py``.  ReportLab primitives keep
the output as editable/vector text without requiring a TeX installation.

Run from the repository root:
    python assets/paper_diagrams/architecture/architecture_pdf.py
"""

from __future__ import annotations

from pathlib import Path
import math

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import inch


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "indicon_gate_paper" / "figures" / "gdn_architecture.pdf"
W, H = 7.16 * inch, 4.40 * inch  # design coordinates for the full-width figure
CROP_BOTTOM = 60  # remove the unused white band below the dashed block boundary
PAGE_H = H - CROP_BOTTOM

INK = HexColor("#243447")
BLUE, BLUE_EDGE = HexColor("#DCEEFF"), HexColor("#2B6CB0")
GREEN, GREEN_EDGE = HexColor("#E2F5E9"), HexColor("#2F855A")
ORANGE, ORANGE_EDGE = HexColor("#FFF0D6"), HexColor("#C05621")
PURPLE, PURPLE_EDGE = HexColor("#EEE6FF"), HexColor("#6B46C1")
GRAY, GRAY_EDGE = HexColor("#F3F4F6"), HexColor("#7B8794")


def rounded_box(c: Canvas, x: float, y: float, w: float, h: float, fill, edge, title: str,
               lines: tuple[str, ...] = (), *, title_size: float = 7.4,
               body_size: float = 6.0, radius: float = 4.0,
               leading: float | None = None) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(edge)
    c.setLineWidth(1.1)
    c.roundRect(x, y, w, h, radius, stroke=1, fill=1)
    c.setFillColor(INK)
    c.setFont("Times-Bold", title_size)
    c.drawCentredString(x + w / 2, y + h - title_size - 3.0, title)
    c.setFont("Times-Roman", body_size)
    line_h = leading if leading is not None else body_size + 2.0
    yy = y + h - title_size - 8.5 - body_size
    for line in lines:
        c.drawCentredString(x + w / 2, yy, line)
        yy -= line_h


def label(c: Canvas, x: float, y: float, text: str, *, size: float = 6.0,
          bold: bool = False, align: str = "center", color=INK) -> None:
    c.setFillColor(color)
    c.setFont("Times-Bold" if bold else "Times-Roman", size)
    if align == "left":
        c.drawString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawCentredString(x, y, text)


def arrow(c: Canvas, x1: float, y1: float, x2: float, y2: float, *, color=INK,
          width: float = 1.05, head: float = 4.2) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(width)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    p = c.beginPath()
    p.moveTo(x2, y2)
    for delta in (math.pi * 0.82, -math.pi * 0.82):
        p.lineTo(x2 + head * math.cos(angle + delta), y2 + head * math.sin(angle + delta))
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def dashed_boundary(c: Canvas, x: float, y: float, w: float, h: float, title: str) -> None:
    c.setDash(4, 3)
    c.setStrokeColor(BLUE_EDGE)
    c.setLineWidth(1.0)
    c.roundRect(x, y, w, h, 6, stroke=1, fill=0)
    c.setDash()
    c.setFillColor(colors.white)
    c.setFont("Times-Bold", 7.0)
    tw = c.stringWidth(title, "Times-Bold", 7.0)
    c.rect(x + 7, y + h - 4, tw + 7, 9, stroke=0, fill=1)
    label(c, x + 10, y + h - 2, title, size=7.0, bold=True, align="left", color=BLUE_EDGE)


def make_figure(output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    c = Canvas(str(output), pagesize=(W, PAGE_H), pageCompression=1)
    # Keep all existing design coordinates stable while cropping the empty
    # lower margin from the emitted PDF page.
    c.translate(0, -CROP_BOTTOM)
    c.setTitle("Gated DeltaNet architecture used in the small synthetic study")
    margin = 16
    label(c, margin, H - 18, "Architecture used in the controlled synthetic experiments", size=9.5, bold=True, align="left")
    label(c, W - margin, H - 18, "small recipe adaptation", size=6.2, align="right", color=GRAY_EDGE)

    # Top-level model pipeline.
    # Keep the stack close to the expanded block so the full-width figure does
    # not waste its center on a large blank band.
    top_y, bh = H - 88, 37
    boxes = [
        (margin, 69, BLUE, BLUE_EDGE, "Token IDs", ("[B, T]",)),
        (margin + 79, 78, BLUE, BLUE_EDGE, "Word embedding", ("8192 x 128", "learnable")),
        (margin + 169, 118, GREEN, GREEN_EDGE, "4 Transformer blocks", ("sequence mixer = GDN", "state mixer = Identity")),
        (margin + 300, 74, PURPLE, PURPLE_EDGE, "Final LayerNorm", ("d = 128",)),
        (margin + 385, 102, ORANGE, ORANGE_EDGE, "Tied LM head", ("128 -> 8192 logits",)),
    ]
    for x, bw, fill, edge, title, lines in boxes:
        rounded_box(c, x, top_y, bw, bh, fill, edge, title, lines, title_size=7.0, body_size=5.8)
    for i in range(len(boxes) - 1):
        x, bw = boxes[i][0], boxes[i][1]
        nx = boxes[i + 1][0]
        arrow(c, x + bw + 3, top_y + bh / 2, nx - 3, top_y + bh / 2, color=BLUE_EDGE)
    label(c, margin, top_y + bh + 5, "No learned position table (max_position_embeddings = 0)", size=5.7, align="left", color=GRAY_EDGE)
    label(c, W - margin, top_y + bh + 5, "tied input/output weights", size=5.7, align="right", color=GRAY_EDGE)

    # Expanded pinned TransformerBlock: LayerNorm and identity state mixer are
    # explicit because they are part of the actual Zoology backbone.
    bx, by, bw, bhall = margin, 70, W - 2 * margin, 145
    dashed_boundary(c, bx, by, bw, bhall, "one pinned TransformerBlock")
    label(c, bx + 9, by + bhall - 16, "The block keeps the Zoology backbone's LayerNorm and residual structure; the GDN mixer is the controlled architectural variable.", size=5.8, align="left", color=GRAY_EDGE)
    x0, lane_y = bx + 12, by + 87
    rounded_box(c, x0, lane_y, 47, 29, PURPLE, PURPLE_EDGE, "LayerNorm 1", ("d = 128",), title_size=6.4, body_size=5.5)
    rounded_box(c, x0 + 57, lane_y, 54, 29, GREEN, GREEN_EDGE, "RecipeGDN", ("input x_t",), title_size=7.0, body_size=5.6)
    rounded_box(c, x0 + 122, lane_y, 48, 29, ORANGE, ORANGE_EDGE, "Residual", ("+ skip input",), title_size=6.4, body_size=5.5)
    rounded_box(c, x0 + 181, lane_y, 47, 29, PURPLE, PURPLE_EDGE, "LayerNorm 2", ("d = 128",), title_size=6.4, body_size=5.5)
    rounded_box(c, x0 + 238, lane_y, 48, 29, GRAY, GRAY_EDGE, "Identity", ("state mixer",), title_size=6.4, body_size=5.5)
    for a, b in ((x0 + 47, x0 + 57), (x0 + 111, x0 + 122), (x0 + 170, x0 + 181), (x0 + 228, x0 + 238)):
        arrow(c, a + 2, lane_y + 14.5, b - 2, lane_y + 14.5, color=BLUE_EDGE)
    arrow(c, x0 + 286, lane_y + 14.5, x0 + 336, lane_y + 14.5, color=BLUE_EDGE)
    rounded_box(c, x0 + 338, lane_y, 58, 29, ORANGE, ORANGE_EDGE, "Block output", ("to next block",), title_size=6.4, body_size=5.5)

    # Internal GDN detail row.
    # Give the recurrence its own readable area.  It is the conceptual centre
    # of the figure, so the equations are deliberately larger than the
    # implementation annotations around it.
    gy, gx = by + 13, x0 + 57
    detail_h = 62
    rounded_box(c, gx, gy, 55, detail_h, BLUE, BLUE_EDGE, "Q / K / V", ("linear projections", "depthwise conv + SiLU", "Q/K normalized", "V width 256"), title_size=6.7, body_size=5.5)
    rounded_box(c, gx + 64, gy, 63, detail_h, ORANGE, ORANGE_EDGE, "Decay alpha", ("scalar: alpha_t [H, 1]", "channel: alpha_t [H, 128]"), title_size=6.5, body_size=5.0)
    rounded_box(c, gx + 136, gy, 52, detail_h, ORANGE, ORANGE_EDGE, "Write beta", ("sigmoid", "beta_t [H, 1]"), title_size=6.7, body_size=5.1)
    rounded_box(c, gx + 196, gy, 150, detail_h, GREEN, GREEN_EDGE, "Value-by-key recurrence", (
        "D_t = alpha_t I (scalar)",
        "D_t = diag(alpha_t) (channel-wise)",
        "S~_t = S_(t-1) D_t",
        "S_t = S~_t + beta_t (v_t - S~_t k_t) k_t^T",
        "readout: S_t q_t",
    ), title_size=7.0, body_size=7.0, leading=7.8)
    rounded_box(c, gx + 354, gy, 56, detail_h, PURPLE, PURPLE_EDGE, "Output gate", ("SiLU(g_proj)", "norm + o_proj"), title_size=6.7, body_size=5.2)
    # Route both buses through the gaps between the boxes.
    detail_top = gy + detail_h
    source_y = (detail_top + lane_y) / 2
    source_x = gx + 27
    branches = (gx + 27.5, gx + 95.5, gx + 162)
    c.setStrokeColor(GREEN_EDGE)
    c.setLineWidth(0.85)
    c.line(source_x, lane_y, source_x, source_y)
    c.line(source_x, source_y, gx + 382, source_y)
    for branch_x in (*branches, gx + 382):
        arrow(c, branch_x, source_y, branch_x, detail_top + 0.6, color=GREEN_EDGE, width=0.85, head=3.2)

    merge_y = gy - 7
    target_x = gx + 196 + 75
    c.line(branches[0], merge_y, target_x, merge_y)
    for branch_x in branches:
        c.line(branch_x, gy, branch_x, merge_y)
    arrow(c, target_x, merge_y, target_x, gy - 0.6, color=GREEN_EDGE, width=0.85, head=3.0)
    arrow(c, gx + 346, gy + 28, gx + 354 - 1, gy + 28, color=GREEN_EDGE, width=0.85)
    c.showPage()
    c.save()
    return output


if __name__ == "__main__":
    print(make_figure())
