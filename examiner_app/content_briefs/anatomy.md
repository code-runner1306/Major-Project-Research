# Content brief: Anatomy harness (`anatomy`)

Question: does the scalar vs channel-wise decay result hold in the published Anatomy of Associative Recall experiments?
Status: `results` (no checkpoints on this machine; saved results only). Verified from the files listed; result files win over prose.

## Sources and status
- Upstream code: `third_party/dualgoose` (JIBSIL/dualgoose commit 8ca624a; Boesch & Wee, arXiv 2609.16183). Provenance and our patch: `third_party/dualgoose/UPSTREAM.md`.
- Our patch: `dualgoose/mixers.py` `DeltaNetRefLayer(decay="channel" | "scalar")`; new recurrence `gated_deltanet_scalar`; `scripts/mqar_probe.py`, `scripts/long_retrieval_probe.py` arms `armed_gdn`, `armed_gdn_scalar`.
- Launcher: `run_A6000_dualgoose_granularity.ps1` (repo root). Frozen predictions: `third_party/dualgoose/PREDICTIONS_granularity.json` (written before training).
- Analysis: `third_party/dualgoose/scripts/analyze_granularity.py` writes `results/granularity_report.json` and `.md`. The page reads the JSON.
- Checkpoints: none in `third_party/dualgoose` (no `checkpoints/` folder). The try slide is therefore `unavailable`.
- Git provenance of our runs: `results/mqar_probe/*_gran_*.json` record commit `bf82979` on `maink` with `dirty: true` (uncommitted files at run time).

## Cells (same config as the upstream reference tags)
| Cell | Probe | Config | Lock-in threshold | Reference tag (upstream) | Ours |
|---|---|---|---|---|---|
| E1 | `mqar_probe` | MQAR K=32, d=32, 1 layer, 6000 steps, rebinding control | 0.90 | `armed_gdn_k32_d32l1_h10` | `*_k32_d32l1_gran_seed*` |
| E4 | `long_retrieval_probe` | 4 pairs, length 64, 48-token haystack, 2 layers, 3000 steps, curriculum | 0.80 | `armed_gdn_L64p4_d32_h5_d4c` | `*_L64p4_d32_granE4_seed*` |
| E5 | same, `--collision` | as E4, haystack = 24 decoy pairs reusing table keys | 0.80 | `armed_gdn_L64p4_d32_h10ck` | `*_L64p4_d32_granE5_seed*` |
E4 and E5 differ only by the collision switch.

## Arms
- `armed_gdn` = channel-wise decay (our reproduction of the published cell). Plain name on page: "channel-wise decay".
- `armed_gdn_scalar` = one decay value per token shared by all key channels. Plain name: "scalar decay". Our addition.
- Upstream reference arms (published numbers, seeds in `upstream_results/`): `armed_gdn` (channel-wise + short convolution), `armed_gdn_fwd` (forward-only stream), `gated_deltanet` (channel-wise, no convolution), `attention`, `deltanet` (no decay), `mamba2_ref`, etc. Upstream `gated_deltanet` / `armed_gdn` decay is channel-wise (UPSTREAM.md, code at `mixers.py` line ~264: "published cell: channel-wise decay").

## Verified facts
- Decay code: `mixers.py` `w = torch.sigmoid(parts[4]) if self.gated`; `state = state * w[:, ti].unsqueeze(1)` (decay on the key axis; scalar is broadcast). `decay_width = d_key` for channel, `1` for scalar.
- Upstream paper text (`Anatomy Research paper.pdf`, Section 2 eq. 2) writes Gated DeltaNet with a scalar gate. The code is channel-wise (UPSTREAM.md). The page must say so: the published numbers come from the channel-wise cell.
- The paper's own decay comparison (Section 4.2, Table 1): deltanet (no decay) 0.561 vs gated deltanet (decay, no conv) 0.518 at K=32, n=20, p=0.86. "Decay costs nothing measurable" in that comparison. The paper says "fine-grained (channel-wise) gating can improve recall over scalar gating (Kimi Team, 2025)" and that any liability is "specific to the scalar-decay knob".
- Our E1 reproduction: channel-wise mean 0.983 (10 seeds), identical to upstream `armed_gdn_k32_d32l1_h10` (0.983). Lock-in 10/10 both.
- Scalar arm E1: mean 1.000, lock-in 10/10. Exact permutation test on mean difference (channel minus scalar = -0.0168): p = 0.00060 in favour of SCALAR. The report marks G1 "confirmed: false" because the prediction was channel > scalar; it does not say the reverse is significant. Page must state the reverse.
- E4: channel 0.665 (lock-in 6/10), scalar 0.591 (6/10), upstream `armed_gdn` h5_d4c 0.668 (6/10). Fisher p = 1.0 on lock-in counts.
- E5: channel 0.386 (lock-in 3/10), scalar 0.499 (lock-in 5/10), upstream `armed_gdn` h10ck 0.383 (3/10). Fisher p = 0.65. Direction in G3 prediction (channel > scalar) not held.
- Rebinding control (values deranged in the input; a binding reader drops to floor): channel 0.037 / scalar 0.036 at E1; E4 channel 0.038 / scalar 0.020. Floors: E1 value marginal 0.03846, table 0.03125, no-binding 0.06851; E4/E5 value marginal 0.01852.
- Mechanism (E6, `scripts/decay_interventions.py`, run on channel-wise checkpoints only): `original`, `fixed_channel_mean` (each channel's decay fixed to its mean), `token_channel_average` (per-token channel decays replaced by their average), `fixed_scalar_mean` (one constant). Means across seeds (report G4): E1 0.983 / 0.452 / 0.262 / 0.247; E4 0.665 / 0.479 / 0.018 / 0.018; E5 0.386 / 0.217 / 0.019 / 0.019. Saved as per-run JSON in `results/decay_interventions/`.
- Accuracy is pooled over the scored answer slots (`long_retrieval_probe.py`: `correct/total` over `target_mask`). Per-query-position accuracy is NOT saved in any result file. The page must say so.
- Lock-in: "a training run counts as locked in when it reaches the threshold" (glossary `lock-in`, 0.90 for E1, 0.80 for E4/E5).

## Anatomy generator (reused from `examiner_app/server/dataset_generators.py`, `anatomy_e4`/`anatomy_e5`)
- `make_long_retrieval_batch(seq_len=64, pairs=4, n_queries=4, gap=None)`: gap = 48. Layout: table positions 0–7 (keys 5–8 at even positions, values at odd positions), haystack 8–55, queries 56–63 as `[key, MASK]` pairs (MASK = token 63, vocabulary 64).
- Haystack (E4): random values drawn from tokens 9–62, which never includes a key token (5–8).
- Haystack (E5, collision): 24 decoy pairs, decoy keys drawn from the table keys 5–8, decoy values random. The answer is the FIRST (table) binding.
- Scored answers: the 4 masked slots at 57, 59, 61, 63. Dataset-generator notes say the same.
- Glossary term `anatomy-harness` already exists; no new glossary term needed.
- Note on dataset facts: `content_briefs/dataset.md` lists `anatomy_e4` / `anatomy_e5` in the same terms; consistent.

## Conflicts and caveats
- Paper prose (scalar gate) vs code (channel-wise): code wins, UPSTREAM.md documents it.
- `decay_design_results.md` (notes, not an Anatomy result) reports channel-wise better than scalar on TinyStories language models (KDA vs DeltaNet, 10M and 100M). Not verified here; not an input to the Anatomy page.
- `published_mqar/` and `published_overwrite/` (repo root) are checkpoint folders for the project's own recipe (`PUBLISHED_CONFIRMATION.md`), not the Anatomy harness. Not used on this page.
- The granularity report's `verdicts.G1_capacity.confirmed` is false even though the test is significant in the reverse direction.
