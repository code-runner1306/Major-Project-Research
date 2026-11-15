# Pinned Zoology source

Source: https://github.com/HazyResearch/zoology/tree/de4e258784224e09909c257ff3ea040f089ed660

Upstream HazyResearch Zoology is distributed under Apache-2.0; see LICENSE.
Original source paths:

- mqar.py: `_mqar` from zoology/data/associative_recall.py, copied unchanged.
- model.py: zoology/model.py, with dependency imports adapted and positional
  index allocation placed on the input device so CPU qualification is possible.
- attention.py: zoology/mixers/attention.py, with its two einops reshape calls
  routed through dependency-free equivalents in compat.py.
- compat.py: local adapters. StochasticDepth accepts only p=0, the pinned
  figure2 setting; it cannot silently approximate nonzero stochastic depth.

Source equations, block ordering, tied embeddings, initialization and attention
dropout remain as upstream. The recipe wrapper and GDN mixer extensions live
outside this directory. No claim is made that GDN was in the historical paper's
architecture sweep.
