# Layout-fix training study (protocol `layout-fix-v1`)

Plan and predictions:
[research_notes/training_fix_plan_and_positioning_2026-11-17.md](research_notes/training_fix_plan_and_positioning_2026-11-17.md).
The predictions are also written into each session's `study_protocol.json` before training starts.

## Kaggle: two notebooks, each on GPU T4 x2

Each notebook trains two workers in parallel, one per GPU: scalar on GPU 0, channel-wise on GPU 1.
Each session pushes to its own new results branch after every finished model pair.

1. In each notebook, turn on **GPU T4 x2** and **Internet**. Store a GitHub token that can push to
   this repository as a Kaggle secret named `GH_TOKEN`.
2. Run this cell, setting `arms` per notebook:
   - Notebook 1: `"A2_no_block_first,A0_control"`
   - Notebook 2: `"A1_varied_distance,A3_mixed_rewrites"`

```python
import os, pathlib, subprocess, sys
from kaggle_secrets import UserSecretsClient

arms = "A2_no_block_first,A0_control"   # notebook 2: "A1_varied_distance,A3_mixed_rewrites"
repo = pathlib.Path("/kaggle/working/Major-Project-Research")
if not repo.exists():
    subprocess.run(["git", "clone", "--branch", "maink",
                    "https://github.com/code-runner1306/Major-Project-Research.git", str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GH_TOKEN")
subprocess.run(["bash", "run_kaggle_layout_fix.sh", "--arms", arms, "--push-results"], check=True)
```

Use **Save Version → Save & Run All** so the run does not depend on your browser staying open.
Each session trains 12 models in 6 pairs, taking roughly 6 hours on T4 x2. The runner stops admitting
new pairs before the 11-hour budget runs out, and anything it skips is listed as `deferred` in
`session_status.json`.

To check the configuration without training, add `--plan-only`.

## Results branches

The branch name is printed at the start:
`research/layout-fix/layout-fix-v1/<arms>/<session>`, for example
`research/layout-fix/layout-fix-v1/A2_no_block_first+A0_control/20261118T...`.

- **One commit per finished pair**, plus a final commit with `session_status.json`. A notebook that
  times out keeps every pair it completed.
- **Same layout as the local output.** Files sit at
  `output/layout_fix_study/layout-fix-v1/<session>/<arm>/seed_<n>/<variant>/`.
- **Weights-only checkpoints.** No optimizer state, about 6.5 MB each. That is enough for
  evaluation but not for resuming training.
- **Push failures don't stop training.** A failed push is retried at the next pair. If the final
  push fails, the run exits with an error and the results remain in the notebook's `/kaggle/working`.

## After both sessions finish

On the evaluation machine, fetch both branches and bring their files into this checkout:

```bash
git fetch origin "refs/heads/research/layout-fix/*:refs/remotes/origin/research/layout-fix/*"
```

```bash
git restore --source=origin/research/layout-fix/layout-fix-v1/<arms>/<session> --worktree -- output/layout_fix_study
```

Repeat this for the second branch. `git restore --worktree` writes the files without staging them, so
nothing is added to `maink`. Then run the confirmation on fresh data:

```bash
python confirm_block_layout.py --output research_results/layout_fix_confirmation --device cuda --initializations --layout-fix-root output/layout_fix_study/layout-fix-v1 --arms A0_control A1_varied_distance A2_no_block_first A3_mixed_rewrites --seed-offset 1400000 --extra-distances 20 24
```

The two sessions may run on different Kaggle machines. Each run records
`initialization_manifest.json`; compare the `base_parameter_sha256` values for the same seed and
variant across sessions before claiming that arms share their initial weights.
