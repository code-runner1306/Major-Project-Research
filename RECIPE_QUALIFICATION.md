# Published-recipe qualification

The selected source is historical Zoology commit
`de4e258784224e09909c257ff3ea040f089ed660`, specifically
`zoology/experiments/paper/figure2.py`, its generator, model and trainer.
See [the source comparison](research_notes/recipe_selection_2026-11-15.md).
The pinned executable and paper prose disagree on some batch sizes, depth and
warmup. Use the executable as the reproduction authority and record adaptations.

The first cell is MQAR length64, four pairs, vocabulary8192,100k training cases,
3k development cases, fixed order, effective batch512, AdamW/weight decay0.1,
64 epochs and the four-rate sweep. GDN was not a mixer in this historical sweep:
adding scalar/channel-wise GDN and audited V4 overwrite are our extensions.
The current six-layer diagnostic is not this recipe. Published defaults do not
authorize a full run without timing and learnability evidence.

## Current status

- Source selection completed, including paper/code differences.
- Experimental chunked recurrence implemented outside the production registry.
- CPU output/state/gradient, continuation and real-model optimizer checks pass.
- Two-worker CPU pilot completed; GPU speed and concurrency are unverified.
- A qualification/timing notebook is ready to launch below.
- No new Kaggle run has started; browser access remains unavailable.
- No final training budget or production backend has been selected.

## First Kaggle session: implementation pilot

Use Internet and GPU T4x2. Add GH_TOKEN only if publishing results is desired.
This session measures implementations, not task learning. Its resident random
tensors cannot qualify MQAR or overwrite. It uses the existing two-layer GDN
diagnostic backbone to isolate backend timing; it is not the final published
recipe model or the final batch512 training workload.

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
types and both recurrence implementations. It rejects model subclasses whose
forward semantics may differ. GPU five-step qualification compares the combined
chunked/scored-loss path against reference/full-logit training.

The timing waves are reference/full-logit/two workers,
reference/scored-loss/two workers, chunked/scored-loss/two workers, then chunked
three and four workers only if the two-worker pilot improves optimizer time,
has CPU headroom and leaves25% GPU memory reserve for two processes per GPU.
Each worker runs50 measured optimizer steps after5 warmup updates. Scalar and
channel-wise alternate, assigned GPU0/1/0/1. The three-worker mix is two scalar
and one channel-wise, so its throughput must be interpreted with that mix.
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
process group. Publishing occurs afterwards through bounded Git commands and
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
