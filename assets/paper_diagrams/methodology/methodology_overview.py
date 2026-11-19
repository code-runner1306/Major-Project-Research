"""Create the compact camera-ready methodology overview as a vector PDF."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUT = Path(__file__).resolve().parents[3] / "indicon_gate_paper" / "figures" / "methodology_overview.pdf"


def box(ax, x, y, w, h, *, face, edge="#4B5563", lw=0.8, radius=0.025):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=face,
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )
    )


def label(ax, x, y, s, *, size=5.0, weight="normal", ha="center", va="center", color="#263238", linespacing=1.03):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va=va, color=color, linespacing=linespacing, zorder=4)


def arrow(ax, x1, y1, x2, y2, *, color="#4472A8", lw=0.65):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=lw,
            color=color,
            shrinkA=1,
            shrinkB=1,
            zorder=3,
        )
    )


def card(ax, x, y, w, h, heading, body, *, face, edge, body_size=7.0):
    box(ax, x, y, w, h, face=face, edge=edge, lw=0.65, radius=0.02)
    label(ax, x + w / 2, y + h - 0.105, heading, size=9.0, weight="bold", va="top")
    label(ax, x + w / 2, y + h - 0.255, body, size=body_size, va="top", linespacing=1.0)


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Keep the figure compact while giving each cohort a genuinely readable
    # body size at the full paper width.
    fig, ax = plt.subplots(figsize=(7.12, 1.50), dpi=300)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 7.12)
    ax.set_ylim(0, 1.50)
    ax.axis("off")

    # Compact header: the comparison stays visible without historical LM details.
    label(ax, 3.56, 1.42, "METHODOLOGY: SCALAR VS CHANNEL-WISE DECAY", size=9.0, weight="bold")
    label(ax, 3.56, 1.30, "matched backbone and task budget; scalar and channel-wise gates paired within seed", size=6.5)

    # Four evaluation cohorts, with the initialization split made explicit.
    y, h = 0.43, 0.72
    widths = [1.62, 1.62, 1.62, 1.72]
    xs = [0.10, 1.84, 3.58, 5.32]
    card(ax, xs[0], y, widths[0], h, "Fresh baseline", "MQAR + overwrite\n64-token training\n2 inits; paired gates", face="#FBE4D5", edge="#C97A2B")
    card(ax, xs[1], y, widths[1], h, "Original stress", "held-out recall cells\n64-token training\n3 inits; stress protocol", face="#FFF3C9", edge="#C99A2E")
    card(ax, xs[2], y, widths[2], h, "Anatomy harness", "E1 / E4 / E5\n10 paired seeds/cell\npublished fixed-state tests", face="#DDEAF7", edge="#4472A8")
    card(ax, xs[3], y, widths[3], h, "Filler extension", "frozen checkpoints\ntrain 64; eval 64/256/1,024\n2 inits × 3 seeds", face="#E9E1F5", edge="#7657A8", body_size=6.7)

    # Minimal protocol band.  The cohort wording is intentional: these are
    # evaluation partitions, not four layers or blocks in the model.
    box(ax, 0.10, 0.10, 6.94, 0.22, face="#EEF1F4", edge="#9AA6B2", lw=0.65, radius=0.018)
    label(ax, 3.57, 0.21, "FOUR EVALUATION COHORTS  |  synthetic training length 64  |  filler cells are frozen evaluations", size=6.4, weight="bold", color="#4B5563")

    fig.savefig(OUT, format="pdf", pad_inches=0.0)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(build())
