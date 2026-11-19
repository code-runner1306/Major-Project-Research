# Methodology figure

`methodology_overview.py` is the authoritative source for the compiled vector PDF at `indicon_gate_paper/figures/methodology_overview.pdf`. It follows the reference camera-ready figure style: restrained pastel panels, compact serif text, and explicit arrows. The compact full-width figure foregrounds four paired comparisons: a fresh baseline with two initialization conditions, the original stress protocol with three initialization conditions, the Anatomy E1/E4/E5 harness with ten paired seeds per cell, and a post hoc 64/256/1024-filler extension with two initialization conditions and three seeds.

Run `python assets/paper_diagrams/methodology/methodology_overview.py` to regenerate the PDF. The older `methodology_diagram.tex` remains as an editable TikZ alternative; the paper-ready artifact requested for insertion is the PDF.

Suggested caption: `Methodology overview. Scalar and channel-wise decay are compared under four paired protocols: a fresh two-initialization baseline, a three-initialization stress evaluation, the ten-seed-per-cell Anatomy harness, and a 64/256/1024-filler frozen-checkpoint extension. Outcomes are kept as separate comparisons.`
