# Recipe selection for the resource-constrained GDN study

## Decision

Use the **Zoology MQAR protocol from the paper**, pinned to the NeurIPS-2024
conference paper and the corresponding historical Zoology code at commit
`de4e258784224e09909c257ff3ea040f089ed660` (`iclr24`). Treat the current
Zoology `main` commit `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` as a useful
implementation reference, not as the historical paper commit. The historical
code is the defensible source for the schedule and benchmark construction; the
current code is useful for checking the newer Gated DeltaNet mixer interface.

The first qualification run should reproduce one fixed, easy-to-audit cell:
`vocab_size=8192`, `input_seq_len=64`, `num_kv_pairs=4`, `num_train_examples=100000`,
`num_test_examples=3000`. For the GDN comparison, use the pinned code's
non-attention block shape: four layers, with the chosen `d_model` and head
count recorded in the manifest. Use the paper's four learning rates
(`1e-4`, `4.6416e-4`, `2.1544e-3`, `1e-2`), AdamW with weight decay `0.1`, and
64 epochs. The paper describes a 10% linear warm-up followed by cosine decay;
the pinned historical trainer actually implements AdamW followed by
`CosineAnnealingLR` and has no warm-up. The executable reproduction below
therefore records `warmup=none` unless we deliberately add and validate a
warm-up adapter. Keep the data order fixed. Select the rate using development
data only and evaluate one untouched confirmation set afterwards.

This is a **protocol control**, not yet the final factorial run. It uses the
same recipe for the scalar and channel-wise GDN variants. The overwrite task is
our extension: use the repository's audited V4 generator and the same training
schedule, batch policy, model depth/width, optimizer, and data exposure. Its
generator audit and confirmation split remain separate from MQAR. Any change
from the MQAR generator must be recorded as an extension rather than called a
literal Zoology reproduction.

## Why this recipe

The paper's Appendix J.1 states that MQAR uses AdamW, weight decay 0.1, a 10%
linear warm-up, a four-point `np.logspace(-4, -2, 4)` learning-rate sweep, 64
epochs, 100,000 training examples, and 3,000 evaluation examples. Its global
batch schedule is 512 for `L < 128`, 256 for `L >= 128`, 128 for `L >= 256`,
and 64 for `L >= 512`. The paper describes two-layer GPT-like blocks and
positional embeddings for attention/RNN mixers, but no positional embeddings
for SSM mixers. These are paper-level claims; the executable historical code
must be checked separately before calling a run a reproduction.

The historical `figure2.py` in the pinned code confirms `VOCAB_SIZE=8192`, the
`(L, KV)=(64,4),(128,8),(256,16),(512,64)` cells, 100,000/3,000 examples,
four LR values, and `max_epochs=64`. Its batch selection is 512 for `L=64`
and `L=128`, 256 for `L=256`, and 128 for `L=512`. Its attention cell uses
one-head MHA, dropout 0.1, two layers, and learned positions. Its
non-attention cells use four layers and no absolute position table. All use
`state_mixer=Identity`. The pinned trainer uses AdamW and `CosineAnnealingLR`
without warm-up. This is an explicit paper/code disagreement; the protocol
must choose one authority and record it rather than mixing paper prose with
code behavior.

The current Zoology `main` configuration is a newer sweep, not the historical
recipe: it uses multiple train/test cells, a 256 batch, two-layer hybrid
BaseConv-plus-mixer blocks, two attention heads, no absolute positions for the
hybrid, and 32 epochs. It should not be mixed into the historical control one
field at a time.

## What differs from the failed diagnostic

The completed diagnostic used six layers of width 160, a pure attention control
with learned absolute positions, 10,000 shuffled examples, effective batch 32,
2,000 optimizer steps, and a OneCycle schedule. It therefore had a different
depth, data exposure, batch, schedule, and mixer construction. Its attention
control also did not include the historical local convolution used in the
authors' newer hybrid configurations. Its result cannot be treated as a
failed reproduction of Zoology.

The local convolution distinction must also be stated precisely. The pinned
historical `figure2.py` attention cell is pure MHA inside the standard block;
it does not contain the newer `BaseConv` hybrid. The current Zoology factory
does create attention and Gated DeltaNet models as a `Hybrid` containing a
short `BaseConv` mixer plus the selected sequence mixer. The official NVIDIA
Gated DeltaNet implementation also includes short convolutions in its block.
Therefore a pure-attention diagnostic is not a reproduction of the current
hybrid recipe. The historical pure-MHA control is legitimate when its published
backbone, positions, training recipe and data are reproduced. Its depth and
position mechanism differ from recurrent models and must be reported; it is a
positive control, not a gate-only ablation. Scalar and channel-wise GDN must
share their backbone and all common settings for the gate comparison.

## Resource-constrained adaptation

The complete four-rate, 64-epoch, 100,000-example protocol is not assumed to
fit in one Kaggle session. First run a timing pilot on one rate and one cell.
Measure setup, data generation, warm-up, optimizer steps, validation, final
confirmation evaluation, checkpoint writing, and evidence export separately.
Then choose the largest common budget that leaves a reserve inside 12 hours.
Any reduced budget must be named an adaptation and must be tested for
learnability before the factorial study.

The reference generator allocates dense NumPy arrays for all examples and
sequence positions. At 100,000 examples and `L=64`, this is a material setup
cost on Kaggle. A row-wise or memory-mapped implementation is acceptable only
if it produces equivalent manifests, seeded examples, labels, and query-gap
distributions; otherwise it is a separately named data-loader adaptation.

Use two concurrent workers per Kaggle notebook only after measuring that they
share the two T4 GPUs without CPU throttling or memory pressure. A second
notebook may run the other task family. Do not infer that three or four workers
increase throughput without a measured pilot.

The final scalar-versus-channel-wise study should not start until the adapted
protocol meets predeclared development criteria on both tasks and a fresh
confirmation set. A successful attention control alone does not qualify GDN;
it only proves the task and loss are learnable under that protocol.

## Primary sources

- [Zoology: Measuring and Improving Recall in Efficient Language Models](https://proceedings.neurips.cc/paper_files/paper/2024/file/f271a36160097fbdb06a9adeb1605343-Paper-Conference.pdf), Appendix J.1 (MQAR training protocol).
- [Pinned historical Zoology source](https://github.com/HazyResearch/zoology/tree/de4e258784224e09909c257ff3ea040f089ed660), especially `zoology/experiments/paper/figure2.py`, `zoology/train.py`, and `zoology/model.py`.
- [Current Zoology MQAR configuration](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/experiments/mqar_example_configs/original_mqar_configs.py), used only as an implementation reference for newer hybrid GDN models.
- [Current Zoology model factory](https://github.com/HazyResearch/zoology/blob/1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb/zoology/experiments/models_repo.py), showing the BaseConv-plus-mixer construction and Gated DeltaNet entry.
- [Gated Delta Networks](https://arxiv.org/html/2412.06464v1), Sections 3–4 and Appendix B, for the role of short convolution, output gating, normalization, and the much larger language-model training regime.
