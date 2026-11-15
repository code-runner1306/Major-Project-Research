# Official GDN reference

Pinned source: https://github.com/fla-org/flash-linear-attention/tree/2705ab8ee9a9086bcfa6d079858e4281e01215a1

The `.reference` files are unmodified source snapshots, retained for review;
they are not imported or executed. The upstream MIT license is in LICENSE.
`recipe_model.py` adapts the documented gate initialization and layer operations
to pure PyTorch, adding key-channel decay and a portable chunked executor.
Using the Zoology backbone, width128, one head and expand_v2 is a synthetic
benchmark adaptation, not a reproduction of billion-parameter pretraining.
