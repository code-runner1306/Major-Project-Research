"""Generate the compact vector results figure used by the IEEE paper.

Both panels show paired accuracy differences from the original run artifacts.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


OUT = "indicon_gate_paper/figures/cross_task_results.pdf"
W = 7.16 * inch
H = 3.40 * inch

INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#4B5563")
GRID = colors.HexColor("#D1D5DB")
PANEL = colors.HexColor("#F8FAFC")
SCALAR = colors.HexColor("#1F4E79")
CHANNEL = colors.HexColor("#C55A11")
GREEN = colors.HexColor("#2F6B4F")


def txt(c, x, y, s, size=7.4, font="Times-Roman", color=INK, anchor="left"):
    c.setFont(font, size)
    c.setFillColor(color)
    if anchor == "right":
        c.drawRightString(x, y, s)
    elif anchor == "center":
        c.drawCentredString(x, y, s)
    else:
        c.drawString(x, y, s)


def panel(c, x, y, w, h):
    c.setFillColor(PANEL)
    c.setStrokeColor(GRID)
    c.setLineWidth(0.55)
    c.roundRect(x, y, w, h, 3, fill=1, stroke=1)


def header(c, x, y, title, subtitle):
    txt(c, x, y, title, 9.5, "Times-Bold")
    txt(c, x, y - 12, subtitle, 6.8, "Times-Italic", MUTED)


def synthetic_differences():
    path = Path("research_results/dynamic_decay_boundary_v1/20261117T154211097_be1684fd/metrics.csv")
    metrics = {"mqar": ("MQAR", "exact_query_accuracy", 6),
               "overwrite": ("Overwrite", "latest_value_accuracy", 9)}
    cells = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row["intervention"] != "original":
                continue
            family, metric, _ = metrics[row["family"]]
            key = family, row["initialization"], int(row["seed"]), row["variant"]
            cells[key][row["cell"]] = float(row[metric])
    paired = {}
    for family, _, expected in metrics.values():
        paired[family] = {}
        for init in ("default", "calibrated", "joint"):
            values = []
            for seed in (42, 43, 44):
                scalar = cells[family, init, seed, "scalar"]
                channel = cells[family, init, seed, "channel-wise"]
                if scalar.keys() != channel.keys() or len(scalar) != expected:
                    raise ValueError(f"Incomplete cell pair: {family}, {init}, {seed}")
                values.append(100 * (mean(channel.values()) - mean(scalar.values())))
            paired[family][init] = values
    return paired


def anatomy_differences():
    root = Path("third_party/dualgoose/results")
    patterns = {
        "E1": "mqar_probe/{arm}_k32_d32l1_gran_seed{seed}.json",
        "E4": "long_retrieval/{arm}_L64p4_d32_granE4_seed{seed}.json",
        "E5": "long_retrieval/{arm}_L64p4_d32_granE5_seed{seed}.json",
    }
    data = {}
    for cell, pattern in patterns.items():
        points = []
        for seed in range(10):
            scores = []
            for arm in ("armed_gdn_scalar", "armed_gdn"):
                path = root / pattern.format(arm=arm, seed=seed)
                record = json.loads(path.read_text(encoding="utf-8"))
                if record["seed"] != seed:
                    raise ValueError(f"Unexpected seed in {path}")
                scores.append(record["eval_accuracy"])
            points.append(100 * (scores[1] - scores[0]))
        data[cell] = points
    return data


def xmap(value, lo, hi, left, right):
    return left + (value - lo) / (hi - lo) * (right - left)


def draw_dot_axis(c, left, right, y, lo, hi, ticks, label):
    c.setStrokeColor(MUTED)
    c.setLineWidth(0.5)
    c.line(left, y, right, y)
    for tick in ticks:
        xx = xmap(tick, lo, hi, left, right)
        c.setStrokeColor(GRID)
        c.setLineWidth(0.35)
        c.line(xx, y - 4, xx, y + 92)
        txt(c, xx, y - 12, f"{tick:g}", 5.7, color=MUTED, anchor="center")
    if label:
        txt(c, (left + right) / 2, y - 23, label, 6.1, color=MUTED, anchor="center")


def draw_synthetic_dots(c, x, y, w, h):
    panel(c, x, y, w, h)
    header(c, x + 10, y + h - 16, "A. Synthetic paired differences", "Channel-wise minus scalar, percentage points; one dot per seed / initialization")
    left = x + 56
    right = x + w - 12
    base = y + 28
    values = synthetic_differences()
    init_colors = dict(zip(("default", "calibrated", "joint"), map(colors.HexColor, ("#205D95", "#B65A20", "#416D4B"))))
    means = []
    for family, conditions in values.items():
        points = [value for group in conditions.values() for value in group]
        means.append((family, mean(points), f"{sum(value > 0 for value in points)}/{len(points)}"))
    draw_dot_axis(c, left, right, base, -25, 40, (-20, -10, 0, 10, 20, 30, 40), "")
    zero = xmap(0, -25, 40, left, right)
    c.setStrokeColor(colors.HexColor("#111827"))
    c.setLineWidth(0.8)
    c.line(zero, base - 2, zero, base + 96)
    group_y = {"MQAR": base + 67, "Overwrite": base + 25}
    inits = [("default", -8), ("calibrated", 0), ("joint", 8)]
    for family, mean_value, positive in means:
        cy = group_y[family]
        txt(c, x + 10, cy + 1, family, 7.2, "Times-Bold")
        for init, offset in inits:
            vals = values[family][init]
            for seed_i, val in enumerate(vals):
                xx = xmap(val, -25, 40, left, right)
                yy = cy + offset + (seed_i - 1) * 3.2
                c.setFillColor(init_colors[init])
                c.circle(xx, yy, 2.0, fill=1, stroke=0)
        mx = xmap(mean_value, -25, 40, left, right)
        c.setStrokeColor(INK)
        c.setFillColor(colors.white)
        c.setLineWidth(0.7)
        c.circle(mx, cy + 15, 3.1, fill=1, stroke=1)
        txt(c, right - 1, cy + 14, f"mean {mean_value:+.2f} pp; {positive}", 5.8, color=MUTED, anchor="right")
    for index, (init, color) in enumerate(init_colors.items()):
        cx = x + 16 + 62 * index
        c.setFillColor(color)
        c.circle(cx, y + 8, 2, fill=1, stroke=0)
        txt(c, cx + 5, y + 6, "mean decay" if init == "calibrated" else init, 6.1, color=MUTED)


def draw_anatomy_dots(c, x, y, w, h):
    panel(c, x, y, w, h)
    header(c, x + 10, y + h - 16, "B. Anatomy harness paired differences", "Channel-wise minus scalar accuracy, percentage points; 10 seeds per cell")
    left = x + 42
    right = x + w - 46
    base = y + 27
    draw_dot_axis(c, left, right, base, -100, 100, (-100, -50, 0, 50, 100), "")
    zero = xmap(0, -100, 100, left, right)
    c.setStrokeColor(colors.HexColor("#111827"))
    c.setLineWidth(0.8)
    c.line(zero, base - 2, zero, base + 94)
    data = anatomy_differences()
    means = {cell: mean(points) for cell, points in data.items()}
    rows = [("E1", base + 68), ("E4", base + 42), ("E5", base + 16)]
    for cell, cy in rows:
        txt(c, x + 10, cy - 2, cell, 7.2, "Times-Bold")
        for i, val in enumerate(data[cell]):
            xx = xmap(val, -100, 100, left, right)
            yy = cy + (i - 4.5) * 2.2
            c.setFillColor(SCALAR)
            c.circle(xx, yy, 1.7, fill=1, stroke=0)
        mx = xmap(means[cell], -100, 100, left, right)
        c.setStrokeColor(INK)
        c.setFillColor(colors.white)
        c.setLineWidth(0.7)
        c.circle(mx, cy + 12, 3.0, fill=1, stroke=1)
        txt(c, right + 4, cy + 10, f"{means[cell]:+.2f}", 5.8, color=MUTED)
    txt(c, x + 10, y + 4, "Open circles show the mean difference.", 5.6, color=MUTED)


def main():
    c = canvas.Canvas(OUT, pagesize=(W, H), pageCompression=1)
    c.setTitle("Cross-task scalar versus channel-wise decay results")
    margin = 0.16 * inch
    gap = 0.12 * inch
    inner_w = W - 2 * margin
    top_y = H - margin
    txt(c, margin, top_y - 2, "Scalar versus channel-wise decay: results across tasks and tests", 13.2, "Times-Bold")
    txt(c, margin, top_y - 18, "The measured outcomes are heterogeneous; no tested setting establishes a reliable small-model channel-wise advantage.", 7.6, "Times-Roman", MUTED)
    y = top_y - 0.48 * inch
    top_h = 2.37 * inch
    panel_w = (inner_w - gap) / 2
    draw_synthetic_dots(c, margin, y - top_h, panel_w, top_h)
    draw_anatomy_dots(c, margin + panel_w + gap, y - top_h, panel_w, top_h)
    footer_y = y - top_h - 0.18 * inch
    txt(c, margin, footer_y, "Percentage points. Positive values favor channel-wise decay. Open circles mark descriptive means.", 7.1)
    c.save()


if __name__ == "__main__":
    main()
