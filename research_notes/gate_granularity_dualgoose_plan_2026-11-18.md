# Plan: scalar vs channel-wise GDN decay in the *Anatomy of Associative Recall* harness

Date: 18 November 2026. Status: **plan only**. Nothing has been cloned, built or run.

Source paper: *Anatomy of Associative Recall in Fixed-State Recurrences* (arXiv 2609.16183), file
`Anatomy Research paper.pdf` in the project root. Code, per the paper: https://github.com/JIBSIL/dualgoose.
Appendix B lists the exact command for every table.

## 1. Question and why this harness

**Question:** with everything else held identical, does per-channel decay change Gated DeltaNet's
recall when memory is constrained, and when it must resist distractors?

**Why this harness:**
- **The paper names the gap itself.** It measures only GDN's scalar decay and says any liability it
  finds "is specific to the scalar-decay knob at constrained state", citing channel-wise gating as
  untested here.
- **It shows where to look.** Recurrent cells differ only at constrained state (width 32). At width
  128 "every cell solves every setting", which explains our null granularity results at width 128.
- **Its protocol is published and strong:** 10-seed headline cells, two-sided permutation tests,
  lock-in rates with Fisher exact tests for bimodal seeds, a rebinding control, and floors stated for
  every task.

**Contribution if it works:** the first single-knob, state-matched scalar vs channel-wise GDN
comparison inside a published recall harness, using that harness's own controls.

## 2. Arms (all in the dualgoose harness, all at their published settings)

| Arm | What it is | Role |
|---|---|---|
| `armed_gdn` (theirs) | GDN, scalar decay, short convolution | Scalar baseline, as published |
| `armed_gdn_ch` (**new**) | Identical except decay is per key channel: `a_proj` outputs d_k values per head, with per-channel `A_log`/`dt_bias` | The single changed knob |
| `armed_gdn_ch_fixed` (**new**, evaluation only) | `armed_gdn_ch` checkpoints with each channel's decay replaced by its training-mean constant | Separates channel diversity from input dependence (our earlier tool) |
| `armed_gdn_ch_avg` (**new**, evaluation only) | `armed_gdn_ch` checkpoints with per-token channel decays averaged into one scalar | Removes channel diversity post hoc |
| `gated_deltanet` / `gated_deltanet_ch` | Both gates without the convolution | Secondary: does the gate effect depend on the convolution, as their transition effect did? |

**Implementation rule.** The channel-wise arm copies their GDN cell line for line, and changes only
the decay projection's output size and the decay parameter shapes. Their scalar cell is not edited.

**State and parameter matching.** Recurrent state is the same: decay granularity does not change the
d_k × d_v state. The parameter difference (`a_proj` grows from d×H to d×d_k) is reported; at d=32 it
is about 1k parameters. Run a parameter-matched scalar control (`armed_gdn` with width raised until
the counts match, or with an equal-size unused projection) only if a channel-wise advantage appears.

## 3. Experiments (their commands, with our arms added)

| ID | Task (paper table) | Settings | Seeds | Measures |
|---|---|---|---|---|
| E1 | Masked MQAR K-sweep (Tables 1-2) | d=32, 1 layer, K ∈ {8, 16, 32} | 10 | Accuracy, lock-in ≥ 0.9, rebinding drop |
| E2 | Hardened ablation with rebinding (Table 3) | d=32, K ∈ {16, 32} | 10 | Same, with their `--selftest` first |
| E3 | Haystack wall (Table 4) | 4 pairs, L ∈ {64, 128, 256}, 2 layers | 5 | Sanity: expect both at chance without a curriculum |
| E4 | Haystack + distance curriculum (Table 5) | L=64, 2 layers, dense 4, `--curriculum`, 3000 steps | 10 | Lock-in rate |
| E5 | Collision-key + curriculum (Table 6) | E4 plus `--collision` | 10 | Lock-in rate; errors that return a decoy (later) value |
| E6 | Interventions on E1/E4/E5 channel checkpoints | fixed-mean and per-token-average decay | (reuse) | Accuracy change vs the original |

E3 is a guardrail. If the channel-wise arm escapes the wall *without* a curriculum, that is a major
result on its own and must be rechecked with more seeds before anything else.

## 4. Pre-registered predictions

These are written into the repo before any training run, together with the analysis script.

- **G1 (capacity, E1/E2):** at K=32, channel-wise mean accuracy minus scalar is greater than 0, with a
  two-sided permutation test at p < 0.05 over 10 seeds. Both must pass the rebinding control (rebound
  accuracy near the floors).
- **G2 (distractors, E4):** the channel-wise lock-in rate is at least the scalar rate; Fisher exact
  test reported. No direction is assumed significant: the paper found lock-in to be a lottery.
- **G3 (collision, E5):** the channel-wise lock-in rate is higher than scalar, and among failures the
  share of errors returning a decoy (later) value is lower. This is the condition where per-channel
  decay should matter most, because the first binding must survive later writes to the same key.
- **G4 (mechanism, E6):** if channel-wise beats scalar, fixed-mean decay keeps most of the advantage
  and per-token averaging removes it. This would replicate our MQAR intervention pattern in a new
  harness.
- **Null rule:** if G1-G3 all fail at 10 seeds, the conclusion is "no detectable granularity effect at
  constrained state in this harness". That is a valid result that bounds the claim.

Thresholds are not changed after seeing results. A positive G1 at K=32 alone is reported as a
highest-load effect, not a general advantage.

## 5. Steps

1. **Get permission and fetch (user approval needed).** Clone `JIBSIL/dualgoose` into
   `.scratch/dualgoose/` (third-party code in its own folder), check its license and record the
   commit hash. Read the code before running anything.
2. **Reproduce first.** Run their E1 command for `armed_gdn` and `gated_deltanet` with seeds 0-2 at
   d=32. They must match Table 1/2 within the seed spread (for example armed GDN at K=32 ≈ 0.98, GDN
   without the convolution ≈ 0.52) before any new arm is added. If they don't match, stop and diagnose.
3. **Add the channel-wise arm** as a new model name in their model registry (a small patch, kept in
   our repo as a diff). Unit tests:
   - With all channel decays tied, it gives exactly the scalar arm's output (proves only the knob
     changed).
   - Serial and chunked paths agree.
   - Parameter count is reported.
4. **Freeze predictions.** Commit section 4 and the analysis script before step 5.
5. **Run E1-E5** on the A6000 with their commands plus the new arm names. Run E3 first (cheapest
   sanity check), then E1/E2, then E4/E5.
6. **Run E6** interventions on the saved channel-wise checkpoints (adapt our `analyze_write_gate`
   patching approach).
7. **Analyse:** permutation tests, Fisher tests, rebinding drops, decoy-error shares, and per-seed
   tables. Write a review note in the same style as earlier ones.

## 6. Cost estimate (to be measured in step 2)

- Models are tiny: about 29k parameters at d=32 for E1/E2, and 2-layer d=32 for E4/E5.
- Training runs: E1 3 K × 10 seeds × 2-4 arms (60-120); E2 overlaps with E1; E3 3 lengths × 5 seeds
  × 2 arms (30); E4/E5 10 seeds × 2 arms each (40).
- **Expected total: hours, not days, on the A6000.** Measure the per-run time in step 2 and re-plan if
  it exceeds about 10 GPU-hours.
- Only the arms above are trained. Parameter-matched controls are added only if a positive effect
  appears.

## 7. Risks and how they are handled

| Risk | Handling |
|---|---|
| Repo missing, unrunnable, or license unclear | Stop at step 1 and report; fall back to SelectivBench (public code) or reimplement from Appendix B, labelled as such |
| Their numbers don't reproduce | Stop at step 2; do not add arms to an unvalidated harness |
| The channel-wise patch changes more than the decay | Equivalence unit test in step 3 (tied channels give exactly the scalar arm) |
| Both arms at chance in E4/E5 | Report the lock-in rates as-is; the curriculum shape is theirs, and we do not tune it per arm |
| Bimodal seeds make means misleading | Use lock-in rates and Fisher tests, as the paper does |
| Small parameter difference explains a win | Parameter-matched scalar control, triggered only by a positive result |

## 8. What this adds to the paper

- **A positive result** gives a clean, published-harness answer that per-channel decay helps under
  memory pressure and distractors, and the interventions say whether it is the channel diversity.
- **A null result** bounds the claim: scalar and channel-wise are indistinguishable even where
  recurrent cells are known to differ. That strengthens our "gate design is not the lever, data
  structure is" conclusion.

Either way, it links our layout-shortcut story to an independent harness, and it answers the
"gate granularity" half of the original research question properly.

## 9. Update after inspecting the repository (18 November 2026)

**Repository.** Cloned to `.scratch/dualgoose`, commit `8ca624a`, MIT licence. Dependencies are
numpy, torch, tiktoken and pytest; `mamba-ssm` is optional. Nothing has been run.

### Key finding: the paper's "Gated DeltaNet" is channel-wise in the code

- **What the paper says.** `paper/main.tex` lines 133 and 425 describe GDN as using a "learned scalar
  gate α_t".
- **What the code does.** `dualgoose/mixers.py`, `DeltaNetRefLayer` with `gated=True`, projects **d_key
  decay values per token**: `extra = self.d_key`, then `w = sigmoid(parts[4])`, then
  `state * w.unsqueeze(1)`. That is a per-channel decay on the key axis, in the style of KDA. The
  comment reads "learned diagonal decay (like RWKV-7's w)".
- **What follows:**
  - Every published `gated_deltanet` and `armed_gdn` number in the paper comes from a channel-wise
    decay cell, not a scalar one.
  - Their claim of a "scalar-gate-specific" decay tax is not supported by their own code.
  - **The missing arm is the scalar one, not the channel-wise one.**
- **Decay parameterisation.** The decay is a plain sigmoid of a linear projection with a bias. FLA's
  GDN form, `exp(-exp(A_log) * softplus(a + dt_bias))`, is not used. This is a second difference from
  real GDN; record it as a caveat.
- **Bidirectionality.** All recall arms except `armed_gdn_fwd` run bidirectionally (a forward plus a
  backward stream with a static merge), because the tasks are masked fills. Report this, since our
  earlier models are causal.

### Revised arms

| Arm | Change from their code | Role |
|---|---|---|
| `armed_gdn` (theirs, unchanged) | none (it is channel-wise) | Channel-wise arm, as published |
| `armed_gdn_scalar` (**new**) | decay projection outputs **1** value per token, broadcast over d_key (`extra = 1`) | The single changed knob |
| `gated_deltanet` / `gated_deltanet_scalar` | the same pair without the convolution | Secondary |
| E6 interventions | fixed-mean and per-token-average decay applied to `armed_gdn` | Mechanism |

### Changes to the steps

- **Step 2 is unchanged:** reproducing their numbers now reproduces the channel-wise arm.
- **Step 3 adds the scalar flag.** Unit test: when the decay projection's d_key output rows are tied
  (identical weights and bias), the channel-wise arm equals the scalar arm. The parameter difference
  is small: d_key - 1 output rows of the decay projection.
- **Optional extra arms:** FLA-parameterised decay (`A_log`/`dt_bias`), scalar and channel-wise. These
  connect the result to real GDN and KDA. Add them only after the main comparison, and only if needed.
- **Reporting:** state the paper-vs-code discrepancy plainly. If it holds up, contacting the authors
  is appropriate before publishing a claim about it.

## 10. Implementation status (18 November 2026)

Implemented in `third_party/dualgoose/`. Provenance and every change are listed in `third_party/dualgoose/UPSTREAM.md`.

- **Scalar arm:** `armed_gdn_scalar` (and `gated_deltanet_scalar`, `armed_gdn_fwd_scalar`). It is a single knob:
  the decay projection outputs one value per token instead of `d_key`. The published arms are unchanged.
- **Tests** (`tests/test_scalar_decay.py`, 18 passing with the two upstream test files):
  - the published arms reproduce the authors' recorded parameter counts exactly;
  - tying the channel decays reproduces the scalar cell to within about 1e-8;
  - the parameter difference is exactly 2 × (d_key − 1) × (d_model + 1) per bidirectional mixer.
- **Predictions** are frozen in `third_party/dualgoose/PREDICTIONS_granularity.json` before any scalar run. R0
  (reproduction) was added: G1-G4 are interpreted only if our `armed_gdn` reproduces the published cells.
- **Runs:** E1 (MQAR K=32, upstream `_h10`), E4 (haystack + curriculum, `_h5_d4c`) and E5 (collision + curriculum,
  `_h10ck`). Each has 10 seeds × 2 arms, 60 runs in total. All use the paper's Appendix B settings at fp32, as in the
  published runs, with the rebinding control on. E3 (the wall without a curriculum) and the no-convolution pair are
  not trained by default (`-IncludeNoConv` adds the pair to E1).
- **E6:** `scripts/decay_interventions.py` checks itself: without an intervention it reproduces each run's recorded
  accuracy exactly.
- **Statistics:** `scripts/analyze_granularity.py`. The Fisher test reproduces the paper's reported p-values (7/10 vs
  1/10 gives 0.0198; 6/10 vs 1/10 gives 0.057).
- **Launcher:** `run_A6000_dualgoose_granularity.ps1`. It trains E1, E4 and E5 concurrently, then runs E6 and the
  analysis.
- **Runtime has not been measured.** The authors report about 115k tokens/s on a GTX 1070, about 16 minutes per E1
  run there. Check the first `DONE` lines in `third_party/dualgoose/logs/` for the real A6000 rate.
- **Still to do:** contacting the authors about the paper-versus-code mismatch, after our results confirm it
  matters.
