# Current methodology: model capacity and calibrated decay

Updated 16 November 2026. This document describes the **current published-recipe
adaptation**, protocol `matched-decay-v1`. It supersedes the nominal 10M/30M scope
in the historical proposal and implementation plan for this experiment. Those
earlier documents and run labels remain provenance, not current parameter counts.

## Research question and scope

We test whether the scalar-versus-channel-wise gating difference changes when
their initial mean decay is matched, and whether that interaction differs between
association retrieval (MQAR) and overwrite with unrelated-association retention.
The current scope is two tasks, four conditions (default/calibrated crossed with
scalar/channel-wise), and configurable seeds. This is a small-model behavioral
study, not a reproduction of a production-scale GDN or evidence about 10M models.

## What model size means

Model size is the number of **unique learned scalar parameters actually
instantiated**, not a configuration label, checkpoint file size, optimizer-step
count, or number of recurrent-state elements. Report total and trainable counts:

```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

PyTorch deduplicates tied parameters through `parameters()`. The embedding and
output head share one weight tensor and must be counted once. Count dense learned
weights even when a particular minibatch does not use every vocabulary entry.
Recurrent state and activations are runtime storage, not extra learned parameters.

Current recurrent architecture: vocabulary 8192, embedding/residual width 128,
four layers, one recurrent head, key width 128, value width 256, depthwise causal
convolution kernel 4, tied embedding/output weights, and no learned position table.
The backbone's state mixer is Identity; there is **no legacy SwiGLU FFN** here.
Each layer retains its two learned LayerNorms and the mixer output RMS weight.

| Component, summed across the model | Scalar | Channel-wise |
|---|---:|---:|
| Shared embedding / tied output:8192 x128 |1,048,576|1,048,576|
| Q,K,V,output-gate,output projections |524,288|524,288|
| Depthwise Q/K/V convolutions |8,192|8,192|
| Decay projection (`a_proj`) |512|65,536|
| Write projection (`b_proj`) |512|512|
| Decay rates and offsets (`A_log`, `dt_bias`) |8|1,024|
| Mixer output RMS weights |1,024|1,024|
| LayerNorms within the four blocks |2,048|2,048|
| Final LayerNorm |256|256|
| **Total parameters** |**1,585,416**|**1,651,456**|
| **Trainable parameters** |**1,585,416**|**1,651,456**|

For this specific one-head architecture, let V be vocabulary size, d be residual
width (and key width), and L be layer count; value width is 2d and convolution
kernel size is 4. The exact counts are:

```text
P_scalar  = V*d + 2*d + L*(8*d*d + 24*d + 2)
P_channel = V*d + 2*d + L*(9*d*d + 25*d)
```

The extra channel parameters are the additional decay-projection rows and
per-channel rates/offsets: `L*(d*d + d - 2) = 66,040`, approximately 4.17% above
the scalar count. Calibration changes neither count. No parameters are frozen.
This is a **same-backbone comparison**, not an exactly parameter-matched one;
capacity overhead is a limitation when interpreting gate granularity effects.

## Why this capacity was chosen

We fixed a resource-feasible architecture rather than solving for a 10M target.
The historical Zoology executable motivated the four-layer non-attention backbone,
8192-symbol vocabulary, and the length 64/four-association task cell. GDN is our
mixer extension; width 128, one head, and value expansion 2 define the selected
small-model adaptation. These choices must not be described as a literal
paper-reproduced 10M GDN configuration. Source identities and paper/code differences
are recorded in [the recipe-selection note](research_notes/recipe_selection_2026-11-15.md).

The practical basis is **measured cost plus demonstrated learnability**: the
current default models trained for 64 epochs / 12,544 steps in approximately 3.9–4.0 hours
per two-worker T4 session, with high development and fresh confirmation accuracy.
That supports using this capacity for the proposed controlled experiment. It
does not establish an optimal model size or isolate which joint recipe change
fixed the earlier failures. Larger-capacity or scaling claims require additional
experiments. We did not choose 128 by a completed width/parameter-count sweep.

The old `10M` preset in `config.py` is six layers of width 160, four heads and
SwiGLU width 426, originally paired with the default vocabulary 50,257. Replacing
that vocabulary by 8192 removes `(50,257-8192)*160 = 6,730,400` embedding parameters.
The saved V3 synthetic runs actually contain 3,329,784 scalar or 3,480,480
channel-wise parameters, despite their `10M` path labels. The current 1.59/1.65M
architecture is a separate later adaptation. None of these reductions comes from
training only a subset of a hidden 10M model.

An approximately 19–20 MB current checkpoint includes FP32 model weights plus
Adam optimizer states and metadata. The learned weights alone occupy about
6.34–6.61 MB in decimal units. Checkpoint bytes therefore do not measure parameter
count directly. Vocabulary size affects total capacity through the embedding;
increasing vocabulary alone adds parameters without increasing recurrent-state
width or the depth of the memory computation.

## Decay targets are not literal offset values

The new mixer computes initial input-dependent decay as:

```text
raw(x) = a_proj(x)
A      = exp(A_log)
alpha(x) = exp(-A * softplus(raw(x) + dt_bias))
```

`dt_bias` is the learned **offset**. The numbers 16, 32, 64, 128 are **target
half-life diagnostics**, not offsets copied into `dt_bias`. In the default
initialization, A is sampled uniformly over (0, 16), with a numerical lower clamp;
dt is sampled log-uniformly between 0.001 and 0.1, then mapped through inverse
softplus to initialize the offset. That default does not guarantee a prescribed
mean alpha on actual training activations.

For the calibrated condition we fit only the existing offset so that, separately
for each gate, the empirical arithmetic mean on calibration tokens satisfies:

```text
mean_training_tokens[alpha(x)] = 2**(-1/h)
```

| Layer, input to output | Target h (tokens) | Target mean alpha |
|---|---:|---:|
|1|16|0.95760328|
|2|32|0.97857206|
|3|64|0.98922801|
|4|128|0.99459942|

Scalar gating fits one offset per layer; channel-wise gating fits 128 offsets per
layer, each to the **same layer target**. Actual fitted offsets depend on the
seed, rate A, projection weights and training activations. They are saved in each
run's `calibration.json`; there is no scientifically correct fixed numerical
offset shared by every model. A and all projection/backbone weights are unchanged
by calibration. Offsets remain trainable afterward, with the existing optimizer
weight-decay exemptions preserved.

The half-life interpretation follows `alpha**h = 1/2` for constant isolated decay.
Matching mean alpha does **not** imply that the product of varying decays has
exactly that half-life. Delta-rule writes, interference and readout further affect
usable memory. We therefore call these initial mean-decay targets, not measured
end-to-end memory lifetimes or guarantees of retrieval at those distances.

## Basis and limits of the target range

The range 16–128 was chosen prospectively as a compact logarithmic range, doubling
between targets, and preserves the earlier exploratory calibration range. The
new one-head model assigns that range across its **four layers**, unlike the
earlier multi-head allocation. This is our hypothesis-driven treatment, not a
claim that a published recipe established these four values as optimal.

Targets 16/32/64 provide persistence scales within the current 64-token context;
128 supplies a slower-decay endpoint beyond that context. This gives a range of
initial forgetting strengths, but the current experiment cannot establish 128-token
retrieval or long-context generalization. Allocating larger targets to deeper
layers is a fixed design choice, not an independently validated layer-order
optimum. Target-range and layer-order sweeps are outside the current study.

The aim is to remove a **difference in initial mean retention** as an explanation
for the scalar/channel comparison. We do not give channel-wise models a more
diverse set of target half-lives within a layer. Calibration still changes
input-dependent decay variation and its local derivatives; matching the mean is
not matching the complete gate distribution or every possible initialization
confound. All claims should refer to the defined initialization treatment.

## Calibration procedure and reproducibility

Use the first 256 training examples (16,384 input tokens at length 64), uniformly
weighting every emitted input position. There is no padding in this fixed-length
cell; filler positions are intentionally included. Training inputs can contain
earlier training answers as specified by the training task; calibration does not
read labels or consume evaluation data. Neither development nor confirmation
examples are used to fit offsets.

Disable dropout, then process layers sequentially, collecting each layer's input
projection after earlier layers have been calibrated. Fit each offset using
float64 bisection in [-80, 80], at most 80 iterations, requiring maximum absolute
mean-decay error <= 1e-6 after offset assignment. An unbracketed or failed fit stops
the run and restores previous offsets; targets are not silently adjusted.
Restore RNG and training modes afterward. The first 256 examples are a fixed,
inexpensive calibration sample, not a demonstrated statistically optimal size.

Persist source/configuration identities, training-token hashes, fitted offsets,
achieved means and variation, calibration timing, unchanged-parameter checks and
shared-initialization hashes. Reference/chunked numerical checks and calibrated
optimizer-trajectory checks precede accepting GPU qualification. A 16-epoch pilot
uses the unchanged 64-epoch cosine schedule. Full study runs require matching
successful GPU pilot evidence and use the same 64-epoch budget for every condition.

## Reporting language

Use: **"A same-backbone, approximately 1.6M-parameter GDN study of gate granularity
and training-data-calibrated mean-decay initialization."** Report exact counts
for both variants and disclose the 4.17% overhead. Do not describe the current
experiment as 10M, equal-parameter, a full production-model reproduction, or an
optimal choice of model size/decay targets. The implementation and run instructions
are in [the calibrated study guide](CALIBRATED_RECIPE_STUDY.md).
