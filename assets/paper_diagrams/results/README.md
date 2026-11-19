# Paired results figure

Run python assets/paper_diagrams/results/generate_cross_task_results.py from the repository root. It generates the vector PDF included by the IEEE manuscript.

Panel A reads the original checkpoint rows from the decay-boundary metrics CSV. It averages exact-query accuracy over the six MQAR cells and latest-value accuracy over the nine overwrite cells. Each point is a channel-wise minus scalar difference for one initialization and seed pair. The nine overwrite conditions include six baseline and shift cells and three paired update diagnostics.

Panel B reads final evaluation accuracy from the raw Anatomy runs and pairs scalar and channel-wise results by cell and seed. Means are descriptive. The SVG and standalone TikZ files are earlier figure drafts.
