# Provenance and modifications

- **Upstream:** https://github.com/JIBSIL/dualgoose, commit `8ca624a`, MIT licence (see `LICENSE`). This is the
  code for Boesch and Wee, *Anatomy of Associative Recall in Fixed-State Recurrences*, arXiv 2609.16183.
- **Vendored:** the `dualgoose/` package, `scripts/`, two upstream tests, `pyproject.toml` and `LICENSE`.
- **Reference results:** upstream `results/mqar_probe` and `results/long_retrieval` are kept unchanged in
  `upstream_results/`, for the reproduction check. New runs write to `results/`.

## Discrepancy found in the upstream code

The paper describes Gated DeltaNet with "a learned scalar gate α_t" (`paper/main.tex` lines 133 and 425), and
says any decay liability "is specific to the scalar-decay knob". In the code, `DeltaNetRefLayer` with `gated=True`
projects `d_key` decay values per token and multiplies the state per key channel (`state * w.unsqueeze(1)`). That
is channel-wise, KDA-style decay. Every published `gated_deltanet` / `armed_gdn` number therefore comes from a
channel-wise cell.

Two further differences from FLA's Gated DeltaNet:
- **Decay form.** The decay is `sigmoid(linear(x))`, not `exp(-exp(A_log) * softplus(a + dt_bias))`.
- **Direction.** The recall arms run bidirectionally (a forward and a backward stream with a static merge),
  except `armed_gdn_fwd`.

## Our modifications (all marked `[Major-Project-Research patch]`)

| File | Change | Effect on published behaviour |
|---|---|---|
| `dualgoose/mixers.py` | `DeltaNetRefLayer(decay="channel" or "scalar")`; scalar projects one decay value per token, broadcast over key channels. New recurrence name `gated_deltanet_scalar`. Evaluation-only `decay_override` hook, `None` by default. | None: the default is the published channel-wise cell |
| `dualgoose/model.py` | Accepts `gated_deltanet_scalar` | None |
| `scripts/mqar_probe.py`, `scripts/long_retrieval_probe.py` | New arms `armed_gdn_scalar`, `gated_deltanet_scalar` (and `armed_gdn_fwd_scalar`); optional `--save-checkpoint DIR` | None unless the new flags or arms are used |
| `scripts/decay_interventions.py` (new) | E6: fixed channel mean, per-token channel average and fixed scalar decay on trained channel-wise checkpoints | n/a |
| `scripts/analyze_granularity.py` (new) | Statistics against `PREDICTIONS_granularity.json` | n/a |
| `tests/test_scalar_decay.py` (new) | Published arms match the authors' recorded parameter counts; tied channel decay reproduces the scalar cell (max difference about 1e-8); the parameter difference is exactly the extra decay rows | n/a |
