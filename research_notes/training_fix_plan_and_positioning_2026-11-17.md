# Training-fix plan and literature positioning

Date: 17 November 2026. Follows
[block_layout_confirmation_review_2026-11-17.md](block_layout_confirmation_review_2026-11-17.md). This is
a plan only: no code or training has been done for it yet.

---

## Part 1: Training-fix experiment

### Question

Which property of the training data makes models learn position-based retrieval for block-first
layouts? Does removing that property give key-based lookup everywhere, without hurting anything else?

### What the current data allows

The training generator is v4 overwrite: 4 keys, 1 rewrite, query distance always 8, length 64.

- About a third of examples (666 of 2000 sampled) are block-first: `TTUUU`.
- In these, the latest value always sits in memory slot 1, and the memory ends right before the
  queries.
- With one rewrite, "second write of the run" and "latest write" are the same token.

So three cues coincide in training, and each arm below removes one of them.

| Arm | Change to training data | Cue removed | If this arm fixes it, the cause is |
|---|---|---|---|
| A0 control | none: reuse existing default checkpoints | (none) | (baseline: 4 of 6 models show the shortcut) |
| A1 varied distance | query distance sampled from 8-16 | fixed distance from query to answer | constant position relative to the query |
| A2 block-first removed | reject `post_target_write_count == key_count - 1` | the layout itself | presence of the block-first layout |
| A3 mixed rewrites | rewrite count sampled from 1-3 | "second write" = "latest write" | rewrite-count ambiguity |

All arms use default initialization only, since gating and initialization made no difference.
Each arm is scalar and channel-wise x seeds 42/43/44, giving 6 models per arm and 18 new models.

### Held fixed (matched exposure)

- 100,000 training examples, 16 epochs, 3136 optimizer steps, effective batch 512, microbatch 32,
  FP32.
- LR 0.00215443469 with the original 64-epoch cosine horizon, vocabulary 8192, length 64, 4 keys.
- Two scored answers per example (latest target and one unrelated key) in every arm.
- The same seed-to-backbone initialization as the A0 checkpoints, so each seed pairs across arms.
- Validation: the original one-rewrite distance-8 development set for every arm, plus each arm's own
  development distribution.

**Unavoidable differences, to be reported.** A3 has more write tokens per example. A1 changes how
much filler sits between the memory and the query, and slightly changes the layout mix because
feasibility depends on distance. Record the realised layout and distance histograms of every
training set in its manifest.

**Control reuse check.** The new runner must reproduce the A0 training manifest hash exactly when
run with the A0 settings. If it does, A0 needs no retraining; if not, retrain A0 as well (6 more
models).

### Evaluation after training

Rerun `confirm_block_layout.py` with these changes:
- `--root` pointing at the new runs and a **new seed offset (1400000)**.
- **Additional distance cells at 20 and 24**, because distances 12 and 16 are in-distribution for
  A1.
- The same H1-H5 metrics, computed per arm.
- The original stress cells (8 keys, distance 24, 2-4 rewrites), so other kinds of robustness are
  tracked too.

### Predictions, to freeze in a protocol file before training

- **P1.** For each of A1, A2 and A3: at most 1 of 6 models meets the H1 shortcut criterion
  (block-first at least 20 points below general at 3 rewrites). A0 has 4 of 6.
- **P2.** No arm lowers general-layout accuracy at 2-4 rewrites by more than 3 points relative to
  A0, or training-distribution development accuracy below 99%.
- **P3, the decision table:**
  - A2 fixes it and A1 does not: the layout itself is the cue.
  - A1 fixes it: constant query position is the cue, and A2 may also work through the same route.
  - Only A3 fixes it: models need a training signal that separates "second" from "latest".
  - Nothing fixes it: a bias of the architecture or optimiser. Then the mechanism work (logit lens
    and patching) becomes the priority.

### Statistical honesty

The per-model effect is huge: about 0-10% vs about 100%. The weak point is the frequency comparison:
4/6 vs 0/6 gives Fisher p of about 0.06. Seeds pair across arms, so also report the paired outcome
for seeds 42 and 44, which show the shortcut under A0. If the best arm's result is borderline, add
seeds 45 and 46 to A0 and that arm (8 more models, about 4 hours).

### Cost and order

The joint run measured 12 models in 5h44m with two workers on the A6000, so about 2h50m per 6 models.

| Session | Arms | GPU time |
|---|---|---|
| 1 | A2 + A1 | about 5h45m |
| 2 | A3 (+ A0 retrain if the hash check fails) | about 2h50m (5h45m) |
| after each | layout confirmation | about 5 min |

A2 runs first: it is the cleanest causal test and needs the smallest code change.

### Implementation sketch (not started)

1. `recipe_data.build_data` gets an optional `layout_filter`, `query_distances` and
   `overwrite_counts`. Rejection sampling keeps exactly N examples. The current
   "exactly one rewrite" assertion becomes "store the last stale value".
2. Add a new runner `run_layout_fix_study.py`, cloned from `run_joint_recipe_study.py`. It has a new
   protocol id `layout-fix-v1`, default initialization only, an `--arm` flag, the two-worker GPU
   scheduler, and publishing off.
3. `confirm_block_layout.py` gets `--seed-offset`, `--extra-distances` and a selector for the
   `layout-fix-v1` runs.
4. A CPU smoke run per arm checks the data histograms, the A0 hash identity and a 2-epoch train.
5. A launcher, `run_layout_fix.ps1`, mirrors `run_joint_recipe.ps1`.

---

## Part 2: Positioning in the shortcut-learning literature

### One-sentence position

> We give a fully controlled, pre-registered instance of underspecification in recurrent
> associative memory. Models with near-perfect accuracy implement a **sub-population shortcut**:
> position-based retrieval for one identifiable third of the training distribution, and key-based
> lookup for the rest, inside the same network.

The new element relative to the shortcut literature is that the shortcut is **local**. Most shortcut
papers show a model using a spurious feature instead of the intended one across the whole input
distribution. Here the same checkpoint uses the intended algorithm on most inputs and a shortcut on a
precisely characterised subset. That subset is the one where the two algorithms agree on every
training example.

### Literature map and how we differ

| Strand | Key works | What they establish | Our difference |
|---|---|---|---|
| Shortcut learning | Geirhos et al. 2020, *Shortcut Learning in Deep Neural Networks* | Rules that pass standard benchmarks but fail to transfer | A synthetic case where the shortcut is exactly identified (a memory slot or structural position) and its scope is a known third of the inputs |
| Underspecification | D'Amour et al. 2020 | Equally accurate predictors differ under stress tests; seed alone can decide | Same pattern: 16 of 18 models vs 2 clean models sharing one seed. Gate design and decay initialization do not change it |
| Heuristics in NLP | McCoy, Pavlick and Linzen 2019 (HANS); McCoy, Min and Linzen 2020 (*BERTs of a feather*) | Targeted diagnostic sets expose heuristics; generalisation varies across fine-tuning seeds | Our layout-stratified test is a HANS-style diagnostic for memory models, with pre-registered hypotheses |
| Why shortcuts win | Hermann et al. 2024, *On the Foundations of Shortcut Learning* (availability vs predictivity) | Easily extracted features beat more predictive ones | In block-first layouts, position is perfectly predictive and plausibly more available through the short convolution. Arm A1 manipulates its predictivity directly |
| Data diversity decides the rule | Qin, Saphra and Alvarez-Melis 2024, *Sometimes I am a Tree* | Mixed data gives seed-dependent, inconsistent rules | The same phenomenon within one task, with the triggering sub-population isolated. Our arms test which diversity removes it |
| Multiple mechanisms in one model | Nikankin et al. 2025, *bag of heuristics*; Ortu et al. 2024, *competition of mechanisms* | LMs combine several partial mechanisms | Clean two-mechanism coexistence in a 1.6M model, with an input condition that predicts which runs |
| Brittle algorithmic shortcuts | Liu et al. 2023, *Transformers Learn Shortcuts to Automata* | Shortcut solutions on synthetic tasks fail out of distribution | Recurrent delta-rule memory rather than Transformers, and a layout-conditional rather than global shortcut |
| Associative recall in efficient models | Zoology/MQAR (Arora et al. 2023); *Anatomy of Associative Recall* (arXiv 2609.16183); KATA (2607.17419); *How Linear Attention Remembers* (2609.33093); Das (TMLR) | Capacity, learnability, curricula, overwrite tests, mechanisms | These mostly report aggregate accuracy or learnability. We show perfect aggregate accuracy can certify the wrong procedure, and give a cheap stratified test |

**The closest overlap to handle carefully is *Anatomy of Associative Recall*.** It reports a
**distance curriculum** that rescues associative-recall training in fixed-state recurrences. It also
attributes some toy-baseline failures to position-generalisation artefacts. Arm A1 (varied distance)
is close to that idea. We should present A1 as testing *which cue* causes a known kind of failure,
not as a new curriculum, and cite it prominently. Our difference is that their failures show up as
low accuracy, while ours are invisible at about 100% accuracy and confined to one layout.

### Claims we can defend

1. A layout-conditional shortcut in GDN trained on keyed overwrite, confirmed on fresh data against
   pre-registered criteria (16 of 18 models). It cannot be seen on the training distribution.
2. The shortcut is position-based retrieval: under distance shifts it returns an unrelated key's
   value from a fixed slot. In the same checkpoints it coexists with key-based lookup.
3. Gate granularity (scalar vs channel) and decay initialization (default, mean-alpha, joint) do not
   change it; seed does. This is underspecification, and it reframes our earlier null gating results
   as evidence.
4. *(Pending Part 1.)* The data property responsible, and a training change that removes it.
5. A cheap evaluation method: layout-stratified sets plus rule-agreement scoring. It works for any
   synthetic memory benchmark with structural regularities, such as fixed query distance or layout.

### Claims to avoid

- That we discovered shortcut learning or underspecification.
- That this is specific to GDN or delta-rule memory. We have not tested attention or Mamba-2 on this
  task. Either add one such control (attention trains in minutes, see `attention_control_20261115`)
  or state it as a limitation.
- Anything about language models or large scale.

### Optional strengthening

- **Architecture control.** Train a small attention model and a Mamba-2-style model on the A0 data
  and run the same stratified test. If they show the same shortcut, the finding is about the data;
  if not, it is about the recurrence. Either answer makes a better paper.
- **Generator audit.** Check whether public MQAR-style generators (Zoology) have comparable
  structural regularities. Zoology MQAR places all keys before all queries, for example. Do not
  claim a flaw without testing.

### Before writing related work

Verify each citation against the primary source. Venues and years here come from search summaries:
Geirhos (Nature Machine Intelligence 2020), D'Amour (JMLR 2022), McCoy (ACL 2019; BlackboxNLP 2020),
Hermann (ICLR 2024), Qin (2024 preprint), Nikankin (ICLR 2025), Ortu (ACL 2024), Liu (ICLR 2023).
Read *Anatomy of Associative Recall* in full, because it is the closest overlap.
