# What the published MQAR recipes actually did

## Recommendation

Yes, the next diagnostic should copy the published MQAR training protocol as closely as this repository allows. The current run is not a faithful reproduction of the canonical Zoology protocol, so its failure cannot be compared directly with the paper's reported learning behavior. This is a protocol-control run, not a replacement for the planned gate-granularity study.

## Primary-source protocol

The Zoology paper's synthetic-training appendix specifies: AdamW with weight decay 0.1; 10% linear warmup; a four-point learning-rate sweep over `np.logspace(-4, -2, 4)`; 64 epochs; global batch size 64 for smaller sequence/model settings (16 or 8 at larger settings); exactly two layers; attention models use position embeddings while pure convolution runs do not; and a fixed data order with 100,000 training and 3,000 evaluation examples. See the paper's Appendix E.2, especially the training-protocol bullets ([Zoology paper, Appendix E.2](https://arxiv.org/pdf/2312.04927), lines 1977–2019 in the PDF text view).

The paper also deliberately makes MQAR difficult by using a vocabulary much larger than the model dimension and by varying the number of key/value pairs and interaction distances. Its generator places key/value pairs first, then repeats keys at sampled distances, and evaluates the recalled values. This is materially different from treating one fixed four-key/length-128 dataset as the whole training distribution ([Zoology paper, MQAR generation procedure](https://arxiv.org/pdf/2312.04927), lines 1931–1976).

The current HazyResearch Zoology repository contains two useful reference configurations, and they should not be confused:

* The simple example is intentionally easy: vocabulary 256, sequence length 64, 10,000 training examples, 1,000 test examples, four key/value pairs, and one-head attention with 0.1 dropout and learned positions ([basic.py](https://github.com/HazyResearch/zoology/blob/main/zoology/experiments/basic_examples/basic.py)). The repository README describes this as a simple two-layer Transformer example, not the full paper sweep ([Zoology README](https://github.com/HazyResearch/zoology)).
* The current `original_mqar_configs.py` is a larger sweep: vocabulary 8,192; 180,000 examples per epoch across several sequence/key-count configurations; batch size 256; a four-point `10^-3` to `10^-1.5` LR sweep; and 32 epochs ([current original_mqar_configs.py](https://raw.githubusercontent.com/HazyResearch/zoology/main/zoology/experiments/mqar_example_configs/original_mqar_configs.py)). It builds two-layer hybrid models with a short BaseConv mixer plus the selected mixer. The current model factory sets `max_position_embeddings=0` for these hybrids and uses two heads in the attention/Gated DeltaNet mixers ([models_repo.py](https://raw.githubusercontent.com/HazyResearch/zoology/main/zoology/experiments/models_repo.py)).

These repository files are living configurations and are newer than the 2023 paper. For a historical reproduction, pin the Zoology commit used by the paper rather than silently treating today's `main` branch as the original experiment.

## Why our current run is not the same recipe

The completed diagnostic uses four fixed keys, sequence length 128, vocabulary 8,192, 10,000 shuffled examples, an effective batch of 32, 2,000 optimizer steps (about 6.4 passes over that pool), six layers at width 160, a pure recurrent GDN versus a pure full-attention control with learned absolute positions, and a OneCycle schedule peaking at 0.001. It is a useful controlled positive-control diagnostic, but it differs from the paper on depth, data volume, data mixture, batch size, schedule, training duration, and model construction. The attention control is also not parameter-matched to GDN.

The paper's result therefore does not imply that our current 2,000-step scalar GDN should learn immediately. The most important differences are the much longer exposure (64 epochs and 100k training examples), two-layer synthetic models, and the prescribed LR sweep. A larger vocabulary is not itself the error: the paper intentionally uses large vocabularies in the hard MQAR protocol. The issue is that vocabulary, model width/depth, data distribution, and training schedule must be copied together if we want a meaningful reproduction.

## Concrete next run

Run a small, faithful protocol control before the factorial study:

1. Use the repository's canonical MQAR generator and a pinned Zoology-compatible data manifest. Keep four keys and a fixed sequence length for the first control, but use 100,000 training and 3,000 evaluation examples with constant order.
2. Use two layers, attention with learned positions, and a GDN mixer configured as the comparison mixer. Do not add the repository's six-layer/width-160 backbone to this control.
3. Use AdamW, weight decay 0.1, 10% linear warmup, and the four learning rates `1e-4, 4.64e-4, 2.15e-3, 1e-2`; train for 64 epochs. Record every rate, rather than selecting one rate after looking at the confirmation set.
4. Keep the independent oracle, query-key counterfactual, and key-free shortcut audits. Use a fresh held-out confirmation set only after the rate sweep is complete.
5. If the attention control still does not learn this easy fixed-length setting, stop and debug the data/label alignment or loss path. If attention learns and GDN does not, the result is a meaningful architecture/optimization diagnostic; it is still not proof of a gate-granularity effect.

This protocol control is likely more expensive than the current run. It should be estimated from one short timing pilot first; do not promise it fits a 12-hour Kaggle session until the 64-epoch workload is measured.

