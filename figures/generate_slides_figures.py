"""Generate presentation figures for the SLM codebase slides.

The figures are intentionally schematic and paper-style: clean boxes, restrained
colors, explicit control points, and enough detail to defend the implementation
without turning the slide into a code listing.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PRODUCTION_MODELS  # noqa: E402
from controls import expected_parameter_count  # noqa: E402


OUT_DIR = ROOT / "figures"
FIG_DPI = 300


COLORS = {
    "ink": "#202124",
    "muted": "#5f6670",
    "line": "#9aa5b1",
    "frame": "#c7d0db",
    "blue": "#e8f2fb",
    "blue_edge": "#79a9d1",
    "green": "#e7f4ea",
    "green_edge": "#72a982",
    "yellow": "#fff3cd",
    "yellow_edge": "#d9aa30",
    "red": "#f8d7da",
    "red_edge": "#c96f78",
    "lavender": "#f0eef8",
    "lavender_edge": "#8e7cc3",
    "cream": "#fcfaf5",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": COLORS["line"],
            "text.color": COLORS["ink"],
            "savefig.facecolor": "white",
        }
    )


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str | None = None,
    *,
    face: str = "white",
    edge: str = COLORS["line"],
    lw: float = 1.4,
    title_size: float = 10.5,
    subtitle_size: float = 8.8,
    title_weight: str = "bold",
    radius: float = 0.05,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.58, title, ha="center", va="center", fontsize=title_size, weight=title_weight)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.25,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=COLORS["muted"],
        )
    return patch


def arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "#2368a2", lw: float = 1.6):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def wrapped(ax, x: float, y: float, text: str, width: int, **kwargs) -> None:
    ax.text(x, y, "\n".join(textwrap.wrap(text, width=width)), **kwargs)


def panel(ax, x: float, y: float, w: float, h: float, title: str, *, face: str = "white", edge: str = COLORS["frame"]):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
    )
    ax.text(x + 0.22, y + h - 0.32, title, ha="left", va="center", fontsize=10.8, weight="bold")


def save_figure(fig, stem: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 7.8)
    ax.axis("off")

    flow_x = 0.55
    flow_w = 3.95
    y0 = 6.88
    step_h = 0.5
    gap = 0.25
    steps = [
        ("Token IDs", "batch x context"),
        ("Token embedding", "vocab x d_model; output head tied"),
        ("N x residual blocks", "RMSNorm -> attention/recurrence -> SwiGLU"),
        ("Final RMSNorm", "normalizes hidden states"),
        ("Language-model head", "next-token logits"),
    ]
    for i, (title, subtitle) in enumerate(steps):
        y = y0 - i * (step_h + gap)
        face = COLORS["green"] if i == 0 else COLORS["blue"] if i in {1, 2} else COLORS["cream"]
        edge = COLORS["green_edge"] if i == 0 else COLORS["blue_edge"] if i in {1, 2} else COLORS["line"]
        box(ax, flow_x, y, flow_w, step_h, title, subtitle, face=face, edge=edge)
        if i < len(steps) - 1:
            arrow(ax, (flow_x + flow_w / 2, y - 0.02), (flow_x + flow_w / 2, y - gap + 0.02), color=COLORS["ink"])

    block_x, block_y, block_w, block_h = 4.85, 4.0, 3.3, 2.55
    panel(ax, block_x, block_y, block_w, block_h, "One residual block")
    ax.text(block_x + block_w / 2, block_y + block_h - 0.62, "pre-norm + residual updates", ha="center", fontsize=8.4, color=COLORS["muted"])
    inner = [
        ("RMSNorm", COLORS["cream"], COLORS["line"]),
        ("Gated DeltaNet attention", COLORS["lavender"], COLORS["lavender_edge"]),
        ("RMSNorm", COLORS["cream"], COLORS["line"]),
        ("SwiGLU MLP", COLORS["yellow"], COLORS["yellow_edge"]),
    ]
    ix, iw, ih = block_x + 0.35, block_w - 0.7, 0.36
    for i, (label, face, edge) in enumerate(inner):
        iy = block_y + block_h - 1.08 - i * 0.45
        box(ax, ix, iy, iw, ih, label, None, face=face, edge=edge, title_size=8.8, radius=0.025)
        if i < len(inner) - 1:
            arrow(ax, (ix + iw / 2, iy - 0.01), (ix + iw / 2, iy - 0.15), color=COLORS["line"], lw=1.0)

    gate_x, gate_y, gate_w, gate_h = 8.75, 4.0, 4.0, 2.55
    panel(ax, gate_x, gate_y, gate_w, gate_h, "Controlled gate variants", face=COLORS["cream"], edge="#b8b1a1")
    ax.text(gate_x + 0.28, gate_y + 1.72, "Scalar head-wise", fontsize=9.3, weight="bold")
    ax.text(gate_x + 0.28, gate_y + 1.43, "alpha: (B, T, H, 1)", fontsize=8.9, family="DejaVu Sans Mono")
    ax.text(gate_x + 0.28, gate_y + 0.95, "Channel-wise", fontsize=9.3, weight="bold")
    ax.text(gate_x + 0.28, gate_y + 0.66, "alpha: (B, T, H, D_head)", fontsize=8.9, family="DejaVu Sans Mono")
    ax.plot([gate_x + 0.25, gate_x + gate_w - 0.25], [gate_y + 1.22, gate_y + 1.22], color=COLORS["line"], lw=0.9)
    arrow(ax, (block_x + block_w + 0.15, block_y + 1.42), (gate_x - 0.1, gate_y + 1.05), color="#2368a2")

    code = (
        "alpha_dim = n_head      # scalar\n"
        "alpha_dim = H*d_head    # channel\n"
        "alpha = sigmoid(alpha_proj(x))\n"
        "alpha = alpha.view(\n"
        "    B, T, H, alpha_width)"
    )
    lower_y, lower_h = 0.55, 2.05

    code_x, code_y, code_w = 5.1, lower_y, 3.05
    panel(ax, code_x, code_y, code_w, lower_h, "Implementation switch")
    ax.text(
        code_x + 0.22,
        code_y + 1.27,
        code,
        fontsize=6.6,
        family="DejaVu Sans Mono",
        va="top",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "#f7f7f7", "edgecolor": "#d8dee6"},
    )

    counts_x, counts_y = 0.55, lower_y
    panel(ax, counts_x, counts_y, 4.25, lower_h, "Configured production scales")
    headers = ["Scale", "Layers", "d_model", "Heads", "Scalar", "Channel-wise"]
    col_w = [0.48, 0.55, 0.68, 0.55, 0.72, 0.88]
    table_x = counts_x + 0.2
    table_y = counts_y + 0.24
    x_positions = [table_x]
    for width in col_w[:-1]:
        x_positions.append(x_positions[-1] + width)
    table_w = sum(col_w)
    row_h = 0.24
    for j, header in enumerate(headers):
        ax.text(x_positions[j] + col_w[j] / 2, table_y + 4 * row_h, header, ha="center", fontsize=6.2, weight="bold")
    for r, (scale, cfg) in enumerate(PRODUCTION_MODELS.items()):
        y = table_y + (3 - r) * row_h
        scalar = expected_parameter_count(cfg, "scalar", 50_257) / 1_000_000
        channel = expected_parameter_count(cfg, "channel-wise", 50_257) / 1_000_000
        values = [scale, cfg.n_layer, cfg.d_model, cfg.n_head, f"{scalar:.2f}M", f"{channel:.2f}M"]
        for j, value in enumerate(values):
            ax.text(x_positions[j] + col_w[j] / 2, y, str(value), ha="center", fontsize=6.4, color=COLORS["ink"])
    for r in range(1, 5):
        ax.plot([table_x, table_x + table_w], [table_y + r * row_h - 0.12, table_y + r * row_h - 0.12],
                color="#edf0f3", lw=0.8)

    control_x, control_y = 8.75, lower_y
    panel(ax, control_x, control_y, 4.0, lower_h, "Control guardrails")
    bullets = [
        "Analytical count formula",
        "Runtime construction assertion",
        "Scalar-matched FFN widening",
        "No dummy parameters",
    ]
    for i, item in enumerate(bullets):
        ax.text(control_x + 0.32, control_y + 1.33 - i * 0.35, f"- {item}", fontsize=8.0, color=COLORS["muted"])

    save_figure(fig, "slm_architecture_slide")


def status_figure() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    lanes = [
        ("Model code", "Complete", COLORS["green"], COLORS["green_edge"]),
        ("Experiment pipeline", "Complete", COLORS["green"], COLORS["green_edge"]),
        ("Local verification", "Complete", COLORS["green"], COLORS["green_edge"]),
        ("V100 pilot", "Pending", COLORS["yellow"], COLORS["yellow_edge"]),
        ("Production grid", "Pending", COLORS["yellow"], COLORS["yellow_edge"]),
        ("Final plots/tables", "Pending", COLORS["yellow"], COLORS["yellow_edge"]),
    ]
    x0, y, w, h, gap = 0.65, 6.05, 1.85, 0.78, 0.28
    for i, (label, status, face, edge) in enumerate(lanes):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, label, status, face=face, edge=edge, title_size=8.9, subtitle_size=8.0)
        if i < len(lanes) - 1:
            arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2), color=COLORS["line"], lw=1.2)

    left_x, left_y = 0.65, 2.65
    panel(ax, left_x, left_y, 3.75, 2.45, "Implemented modules")
    implemented = [
        "Scalar Gated DeltaNet variant",
        "Channel-wise Gated DeltaNet variant",
        "Exact parameter accounting",
        "TinyStories/MQAR/overwrite paths",
        "Cache-aware prefill and decode",
        "Gate/latency/memory/energy diagnostics",
        "Manifest/checkpoint lifecycle",
    ]
    for i, item in enumerate(implemented):
        ax.text(left_x + 0.28, left_y + 1.76 - i * 0.24, f"- {item}", fontsize=7.6, color=COLORS["muted"], va="top")

    mid_x, mid_y = 4.75, 2.65
    panel(ax, mid_x, mid_y, 3.35, 2.45, "Verification evidence")
    verification = [
        ("Unit tests", "13 / 13 passing"),
        ("Smoke path", "offline CPU suite"),
        ("Cache check", "decode equivalence"),
        ("Artifact safety", "drift/corrupt rejection"),
    ]
    for i, (k, v) in enumerate(verification):
        ax.text(mid_x + 0.28, mid_y + 1.78 - i * 0.33, k, fontsize=8.0, weight="bold")
        ax.text(mid_x + 1.55, mid_y + 1.78 - i * 0.33, v, fontsize=8.0, color=COLORS["muted"])

    right_x, right_y = 8.45, 2.65
    panel(ax, right_x, right_y, 4.05, 2.45, "Remaining work")
    pending = [
        "Run V100 pilot for fixed budget",
        "Set production max_optimizer_steps",
        "Train paired 10M-100M grid",
        "Aggregate quality/latency/retrieval/energy",
        "Generate final result plots and tables",
    ]
    for i, item in enumerate(pending):
        ax.text(right_x + 0.28, right_y + 1.76 - i * 0.32, f"- {item}", fontsize=8.0, color=COLORS["muted"], va="top")

    ax.text(0.65, 1.68, "Run lifecycle used by every seed", fontsize=11.2, weight="bold")
    life = [
        ("pending", COLORS["cream"], COLORS["line"]),
        ("training", COLORS["blue"], COLORS["blue_edge"]),
        ("trained", COLORS["blue"], COLORS["blue_edge"]),
        ("evaluating", COLORS["lavender"], COLORS["lavender_edge"]),
        ("complete", COLORS["green"], COLORS["green_edge"]),
    ]
    lx, ly, lw, lh, lgap = 0.65, 0.82, 1.55, 0.5, 0.36
    for i, (label, face, edge) in enumerate(life):
        x = lx + i * (lw + lgap)
        box(ax, x, ly, lw, lh, label, None, face=face, edge=edge, title_size=8.9, radius=0.03)
        if i < len(life) - 1:
            arrow(ax, (x + lw, ly + lh / 2), (x + lw + lgap, ly + lh / 2), color=COLORS["line"], lw=1.1)
    fail_x = lx + 1.5 * (lw + lgap)
    box(ax, fail_x, 0.2, 1.55, 0.38, "failed", None, face=COLORS["red"], edge=COLORS["red_edge"], title_size=8.4, radius=0.03)
    arrow(ax, (fail_x + 0.75, ly - 0.02), (fail_x + 0.75, 0.62), color=COLORS["red_edge"], lw=1.0)

    ax.text(10.15, 1.23, "Status statement", fontsize=10.2, weight="bold")
    ax.text(
        10.15,
        0.58,
        "First-submission paths are implemented.\n"
        "Final metrics wait for the\n"
        "V100 production grid.",
        fontsize=8.6,
        color=COLORS["muted"],
        linespacing=1.35,
    )

    save_figure(fig, "implementation_status_slide")


def main() -> None:
    setup_style()
    architecture_figure()
    status_figure()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
