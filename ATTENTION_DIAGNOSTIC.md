# Four-key attention control diagnostic

Run the existing launcher on Kaggle with GPU T4 x2 and Internet enabled:

```bash
FAMILY=mqar bash run_kaggle_fixed_learnability.sh
```

The default is now `DIAGNOSTIC=attention-control`. To reproduce the earlier test,
set `DIAGNOSTIC=varied` before running this shell file.

## Design

1. Audit four-key V4 data for the selected family using independent last-write and
   query-key-change oracles and key-free baselines. The selected task must pass before
   any training starts. Audits cover the development and confirmation sets.
2. Train scalar/default Gated DeltaNet on GPU 0 and the full causal attention
   control on GPU 1 for the selected family. Run the other family in a separate notebook.
   There are never more than two training workers.
3. Each run starts fresh, trains 2,000 optimizer steps at peak LR 0.001, batch 8,
   accumulation 4 and the existing OneCycle schedule. Both architectures use
   identical seeded shuffled training data, held-out splits and budgets.
4. Use 10,000 generated training examples, four keys, sequence length 128,
   distance 8, one rewrite and vocabulary 8,192. The model backbone has six
   layers and width 160; report actual parameter counts, not the nominal 10M label.
5. Every 200 steps, measure teacher-forced and answer-free accuracy on the first
   seen training examples, recent training examples, and 128 development examples.
   Record first/later-query accuracy and latest/unrelated overwrite accuracy.
6. At the final checkpoint only, evaluate 512 confirmation examples with another
   seed/domain. These are not used for checkpoint selection. The model receives
   no oracle answers during answer-free evaluation.

The attention model replaces every recurrent mixer with causal dot-product
attention and adds learned absolute position embeddings. The FFNs, norms, tied
token/output embeddings, depth and width use the same backbone. Its useful
parameter count and position mechanism differ from GDN: this is a learnability
control, not a capacity-matched architecture ablation. One seed and one learning
rate cannot establish an architectural limitation or prove an implementation bug.

The data-audit baseline `final_two_values` cycles two key-free guesses across
all query ordinals. Previously it omitted queries after the second and therefore
incorrectly marked four-key MQAR checks incomplete.

## Time budget

`TIME_BUDGET_SECONDS=39600` caps coordinator time at 11 hours. It stops jobs
10 minutes before that limit and writes partial results. A separate GNU `timeout`
cap stops the launcher at 11h30m with a 30-second forced-stop grace. This keeps
the launched workload below 12 hours under normal Linux process termination.
Notebook provisioning and pip installation happen before the launcher and are
outside its budget. Full 2,000-step completion cannot be guaranteed: slow or
failed jobs are explicitly marked incomplete. A timeout is never a model verdict.

Optional environment overrides:

```bash
FAMILY=mqar STEPS=1000 EVAL_EVERY=200 bash run_kaggle_fixed_learnability.sh
FAMILY=overwrite SEED=43 OUTPUT_ROOT=/kaggle/working/attention_seed43 bash run_kaggle_fixed_learnability.sh
FAMILY=mqar GIT_PUSH=1 bash run_kaggle_fixed_learnability.sh
```

Publishing is optional and off by default. With `GIT_PUSH=1`, use the `GH_TOKEN`
Kaggle secret in the environment or existing safe Git credentials. Results are
exported to a separate `diagnostic/...` branch through an isolated Git repository;
checkpoints are excluded. Raw checkpoints remain in Kaggle output. Use a safe
credential-free origin URL; do not embed tokens in notebook cells or URLs.

## Outputs

- `<family>_audit.json`: complete baseline and oracle results.
- `<family>/<architecture>/settings.json`: exact model and dataset identities.
- `progress.json`: intermediate accuracy observations, saved every validation.
- `training_metrics.json`: losses, optimizer times, clipping, retries and validation history.
- `summary.json`: final development and confirmation results for that worker.
- `session_status.json`: complete/incomplete status and subprocess exit codes.
- `comparison_report.json`: the selected family's two worker summaries and cautious screening findings.
- `<family>_<architecture>.log`: live worker logs.

The report uses predeclared 80% and 10% thresholds as operational screens, not
statistical significance tests. If both models learn, qualify the paper's four
conditions on clean data. If only attention learns, investigate GDN-specific
optimization/implementation/capacity. If neither learns, investigate the shared
setup and budget. Intermediate or incomplete results require inspecting curves.

Use Save Version -> Save & Run All so Kaggle retains the output independently
of the laptop. No production research runs are started just by editing these files.

## Two separate Kaggle sessions

Use GPU T4 x2 and Internet in each notebook. Enable the GH_TOKEN secret.
Change only FAMILY in the following cell: mqar in one notebook, overwrite in the other.
Each runs GDN and attention concurrently on GPU 0 and GPU 1.
Publishing uses separate branches diagnostic/<family>/seed<seed>_<output-folder>.
Default output folders include family and UTC timestamp. Checkpoints are excluded.
Concurrent sessions need four GPUs total and depend on account availability.

```python
FAMILY = "mqar"  # Use "overwrite" in the second notebook.

import os
import pathlib
import subprocess
import sys
from kaggle_secrets import UserSecretsClient

assert FAMILY in ("mqar", "overwrite")
repo = pathlib.Path("/kaggle/working/Major-Project-Research")
if not repo.exists():
    subprocess.run(["git", "clone", "--branch", "maink",
        "https://github.com/code-runner1306/Major-Project-Research.git", str(repo)], check=True)
os.chdir(repo)
subprocess.run(["git", "fetch", "origin", "maink"], check=True)
subprocess.run(["git", "checkout", "--detach", "origin/maink"], check=True)
print("Source revision:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
os.environ.update(PYTHON_BIN=sys.executable, DIAGNOSTIC="attention-control",
    FAMILY=FAMILY, STEPS="2000", EVAL_EVERY="200", LEARNING_RATE="0.001", GIT_PUSH="1")
os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GH_TOKEN")
subprocess.run(["bash", "run_kaggle_fixed_learnability.sh"], check=True)
```

Use Save Version -> Save & Run All in each notebook.
