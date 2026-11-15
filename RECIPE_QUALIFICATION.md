# Published-recipe qualification

The selected source is historical Zoology commit
`de4e258784224e09909c257ff3ea040f089ed660`, specifically
`zoology/experiments/paper/figure2.py`, its generator, model and trainer.
See [the source comparison](research_notes/recipe_selection_2026-11-15.md).
The pinned executable and paper prose disagree on some batch sizes, depth and
warmup. Use the executable as the reproduction authority and record adaptations.

The first cell is MQAR length64, four pairs, vocabulary8192,100k training cases,
3k published development cases, fixed order, effective batch512, AdamW/weight decay0.1,
64 epochs and the four-rate sweep. GDN was not a mixer in this historical sweep:
adding scalar/channel-wise GDN and audited V4 overwrite are our extensions.
The current six-layer diagnostic is not this recipe. Published defaults do not
authorize a full run without timing and learnability evidence.

## Current status

- Source selection completed, including paper/code differences.
- Experimental chunked recurrence implemented outside the production registry.
- CPU output/state/gradient, continuation and real-model optimizer checks pass.
- Published-backbone/data adapters and batched evaluation pass CPU checks.
- Both task families completed paired CPU timing smokes; a guarded CPU development run also completed.
- GPU speed, learnability and concurrency remain unverified.
- A qualification/timing notebook is ready to launch below.
- No new Kaggle run has started; browser access remains unavailable.
- No final training budget or production backend has been selected.

## First Kaggle session: implementation pilot

Use Internet and GPU T4x2. Add GH_TOKEN only if publishing results is desired.
This session measures implementations, not task learning. Its resident random
tensors cannot qualify MQAR or overwrite. It now uses the pinned four-layer
Zoology backbone with the FLA-based scalar/channel-wise GDN mixer, width128,
one head, key width128 and value width256. Microbatch32 with accumulation16
measures effective batch512. It still does not measure real data generation or
actual retrieval evaluation; those are measured in the next stage.

```python
import os, pathlib, subprocess, sys
from kaggle_secrets import UserSecretsClient

repo = pathlib.Path('/kaggle/working/Major-Project-Research')
if not repo.exists():
    subprocess.run(['git','clone','--branch','maink',
        'https://github.com/code-runner1306/Major-Project-Research.git',str(repo)],check=True)
os.chdir(repo)
subprocess.run(['git','fetch','origin','maink'],check=True)
subprocess.run(['git','checkout','--detach','origin/maink'],check=True)
print('Revision:',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
subprocess.run([sys.executable,'-m','pip','install','-r','requirements.txt'],check=True)
os.environ.update(PYTHON_BIN=sys.executable,GIT_PUSH='1',
                  OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.environ['GH_TOKEN']=UserSecretsClient().get_secret('GH_TOKEN')
subprocess.run(['bash','run_kaggle_recipe_pilot.sh'],check=True)
```

The launcher first compares FP32 outputs, states and all six input gradients on
real recurrence shapes, with scalar/channel-wise gates and exact-zero decay.
It then checks five actual AdamW updates against the reference for both variants.
The recurrence error gates are max absolute and normalizedL2 <=2e-4. For the
optimizer trajectory, max coefficient drift <=1e-4 and tensor normalizedL2
<=5e-4 are required. AdamW can amplify roundoff on near-zero gradients: an
initial 2e-5 absolute-only check failed at one coefficient out of1,048,576
(2.48e-5 drift). Both error measures are retained to avoid hiding wider drift.
These thresholds are recorded before GPU qualification. FP16 is not qualified.

The second optimization computes vocabulary scores only at supervised answer
positions. It retains every input token and recurrent update. CPU checks compare
the loss and every parameter gradient with the full-logit loss for both gate
types and both recurrence implementations. The earlier legacy adapter rejects
model subclasses whose forward semantics may differ; the published backbone
has its own tested scored-logit path. GPU five-step qualification compares the combined
chunked/scored-loss path against reference/full-logit training.

The timing waves are reference/full-logit/two workers,
reference/scored-loss/two workers, chunked/scored-loss/two workers, then the
fastest measured combination with three and four workers only if it improves optimizer time,
has CPU headroom and leaves25% GPU memory reserve for two processes per GPU.
Each worker runs50 measured optimizer steps after5 warmup updates. Scalar and
channel-wise alternate, assigned GPU0/1/0/1. The three-worker mix is two scalar
and one channel-wise, so its throughput must be interpreted with that mix.
A subsequent wave is skipped if measured CPU saturation exceeds90%.
CPU worker threads are capped at1. Outputs include all step times, process CPU
time, host utilization samples, peak GPU allocated/reserved memory, setup,
warmup, a512-example forward/loss evaluation and checkpoint timing.

The phase throughput includes imports, setup, warmup, evaluation and checkpoint
writing; also inspect mean optimizer times to distinguish fixed overhead.
Evaluate throughput and memory together; OOM or failed numerical gates never
authorize promotion. This pilot's evaluation timing is not a substitute for the
final retrieval metrics or calibration timing.

The coordinator has a90-minute workload limit; a separate100-minute watchdog
covers startup, numerical checks and stalls. Workers share the watchdog's
process group. The same watchdog also covers publishing through bounded Git commands and
uses `diagnostic/recipe-pilot/<UTC-output-folder>`. Checkpoints stay in Kaggle.

## Evidence required before the full experiment

1. Actual T4 numerical and model-trajectory passes for every promoted setting.
2. Actual end-to-end timing for the selected published-style model, effective
   batch, real generator and training data pipeline. Include data generation,
   calibration, validation, subgroup retrieval evaluation, checkpoints and export.
3. A common fixed budget chosen on development results and conservative measured
   timing, leaving reserve inside12h. An interrupted run is incomplete evidence.
4. Learnability on both scalar and channel-wise models for both tasks. Require
   a predeclared operational target of80% answer accuracy and80% in each relevant
   subgroup: MQAR first/later query; overwrite latest/unrelated. This is a
   qualification screen, not statistical significance.
5. Fresh confirmation data evaluated only after development selection. Passing
   baseline audits alone is not learned task qualification.
6. Frozen generator/model/optimizer/batch/rate-selection/budget/evaluation settings
   and paired fresh seeds for the final factorial study. All deliberate changes
   from the pinned recipe remain visible in the protocol manifest.

Do not start the full study until these items are supported by saved evidence.

## Real-task timing and guarded development

`run_kaggle_published_recipe.sh` runs exactly two workers, scalar on GPU0 and
channel-wise on GPU1. Set FAMILY=mqar or FAMILY=overwrite. Its default is a
50-step real-task timing run, not64 epochs of training. It creates100k fixed-order
training examples and512 development examples; the smaller development set is
an explicit resource adaptation retaining the user's512-example preference.
The new source-pinned MQAR generator is byte-equivalent to the original code
without its giant repeated vocabulary arrays. Overwrite uses audited V4 with65
emitted tokens so its shifted model input is64 tokens, matching MQAR.

The GDN mixer uses activated short convolutions, normalized Q/K, expanded values,
exp-softplus decay with initialized timescales, sigmoid writes and gated output
RMS normalization. This replaces the earlier simplified sigmoid-decay model;
old checkpoints are not compatible. The channel extension changes key-axis
control only. Stable initialization preserves common scalar/channel weights and
per-head base timescales; all differences are visible in the model and manifest.
No calibrated-initialization qualification has yet been performed.

After the implementation pilot succeeds, append this cell to the same notebook
for one task. Run the other task separately, preserving each timing directory:

```python
FAMILY = 'mqar'  # Or 'overwrite'.
# Replace with the successful implementation pilot's printed output directory.
QUALIFICATION_ROOT = '/kaggle/working/recipe_pilot_REPLACE_WITH_ACTUAL_RUN'
os.environ.update(FAMILY=FAMILY, MODE='timing', BACKEND='chunked',
                  QUALIFICATION_ROOT=QUALIFICATION_ROOT, GIT_PUSH='1')
subprocess.run(['bash','run_kaggle_published_recipe.sh'],check=True)
```

Use BACKEND=reference without QUALIFICATION_ROOT if it is the chosen safe path.
A chunked GPU run rejects CPU-only evidence, mismatched GPU/PyTorch identity,
changed source hashes, or missing GPU numerical/model/two-worker passes.

The evaluator batches examples and projects vocabulary logits only at answer
positions. Its answer-free contract is identical to the previous evaluator:
earlier gold answers are replaced with filler before the pass. It does not feed
model predictions back. CPU tests compare metrics with the old evaluator using
an independent keyed oracle and verify the actual erased input tokens.

The real-task timing summary separates data/model/audit setup, each accepted
optimizer step, both evaluation histories and checkpoint writing. It also checks
actual train/development content overlap and identical paired dataset manifests.
No confirmation examples are generated by this runner.

Development runs require MODE=development, an explicit EPOCHS choice and
TIMING_EVIDENCE_ROOT from that family's successful real-task timing pair. The
budget guard rejects source, precision, batch, data-shape or hardware drift and
nonfinite timing samples. It estimates the slower variant using p95 or120% of
mean step time, validation/checkpoint overhead, doubled setup and15m reserve.
Select one common epoch budget across both families after inspecting these
measurements; neither this script nor its default64 ceiling selects that budget.

The coordinator ceiling is11h, with an outer11h30m watchdog covering training
and publishing. Worker/import/startup time is included in coordinator wall time;
notebook provisioning and package installation precede this ceiling. Results go
to separate diagnostic/published-recipe/<family>/<mode>/<UTC-folder> branches.
Checkpoints remain in Kaggle. This is not a production authorization: final
confirmation, calibrated variants, multi-seed qualification and the final lock
are still outstanding.

## Ready-to-import notebook

`notebooks/kaggle_recipe_qualification.ipynb` includes setup, dependency install,
secret loading and the three stage choices. Its timer starts before cloning and
installing dependencies, and setup time is deducted from the remaining session
budget. The launchers apply independent process-group watchdogs, including
publication. The notebook's own subprocess calls also receive remaining-time
limits. Platform provisioning before the first cell is outside this timer, with
30m nominal headroom retained inside the12h session.

For a fresh notebook, the optional QUALIFICATION_BRANCH and TIMING_BRANCH fields
retrieve only the named JSON evidence files from separate remote result branches;
no branch is merged and no checkpoints are downloaded. This avoids relying on
working directories surviving a new Kaggle job. Source, hardware, precision and
budget gates still validate those files. Each stage must be reviewed before
starting the next; importing the notebook does not launch the final experiment.

## Parameter-count disclosure

This published-width128 synthetic setup is a new benchmark protocol, not a
continuation of the earlier nominal10M/six-layer runs. Actual parameters are
1,585,416 scalar GDN,1,651,456 channel-wise GDN, and1,190,144 attention control.
The control has its published two-layer depth; GDN uses the historical
non-attention four-layer backbone. Do not combine old checkpoints or curves
with this experiment. Parameter counts, state dimensions and initialization
changes must be reported in the eventual paper.
