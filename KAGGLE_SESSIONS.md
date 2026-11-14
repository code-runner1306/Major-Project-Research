# Two-model Kaggle sessions

One seed contains eight models: MQAR and overwrite for C1, C3, C4 and C6.
Use one notebook per condition, with two T4 GPUs in each notebook. Each session
runs exactly two training processes, one on each GPU, then evaluates both saved
models in parallel. Each selected model uses the committed corrected V4 study
config, including its learning rate, 1,000-step budget and validation settings.
There is no additional timing pilot before production training.

For seed 42, run one command in each of four separate notebooks:

```bash
bash run_kaggle_12h_replication.sh --condition C1 --seed 42 --approved-by "Mayank Mehta" --push-results
bash run_kaggle_12h_replication.sh --condition C3 --seed 42 --approved-by "Mayank Mehta" --push-results
bash run_kaggle_12h_replication.sh --condition C4 --seed 42 --approved-by "Mayank Mehta" --push-results
bash run_kaggle_12h_replication.sh --condition C6 --seed 42 --approved-by "Mayank Mehta" --push-results
```

These are four independent sessions, not eight workers inside a single notebook.
Kaggle decides how many GPU notebooks your account can run simultaneously. Queue
the sessions if all four cannot start together.

To choose any two models explicitly:

```bash
bash run_kaggle_12h_replication.sh --models mqar:C1 overwrite:C4 --seed 42 --approved-by "Mayank Mehta" --push-results
```

Available condition IDs: C1 scalar/default, C3 channel-wise/default, C4
scalar/calibrated-geometric, C6 channel-wise/calibrated-geometric. Duplicate model
selections are rejected. The seed must be explicit; all four sessions should use
the same seed for a paired comparison. No production run is started by editing
the launcher or by `--dry-run`.

Preview the task configs without GPUs or Git credentials:

```bash
bash run_kaggle_12h_replication.sh --condition C4 --seed 42 --dry-run
```

The launcher records `--approved-by` and the UTC launch timestamp as authorization
for exactly the two selected models in the session output. It does not approve
the full five-seed study globally.

## Notebook setup and saved cloud execution

Clone branch `maink`, install `requirements.txt`, and enable **GPU T4 x2** and
Internet in notebook settings. Store a GitHub token that can push to your repo
as a Kaggle secret named `GH_TOKEN`; do not paste the token into a cell or URL.
For example, use this cell in each notebook, changing `condition` as needed:

```python
import os, pathlib, subprocess, sys
from kaggle_secrets import UserSecretsClient

condition = "C4"
seed = 42
repo = pathlib.Path("/kaggle/working/Major-Project-Research")
if not repo.exists():
    subprocess.run(["git", "clone", "--branch", "maink",
                    "https://github.com/code-runner1306/Major-Project-Research.git",
                    str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
os.environ["GH_TOKEN"] = UserSecretsClient().get_secret("GH_TOKEN")
subprocess.run(["bash", "run_kaggle_12h_replication.sh", "--condition", condition,
                "--seed", str(seed), "--approved-by", "Mayank Mehta",
                "--push-results"], check=True)
```

Use **Save Version → Save & Run All** for execution independent of your laptop
and for Kaggle to retain the output artifacts. A draft session alone does not
preserve files after its VM is discarded.

## Results and time limit

- Raw output: `/kaggle/working/seed.../`, including checkpoints.
- Git-ready evidence: sibling `seed..._results/`, with metrics, validation
  history, evaluation outputs, manifests, configs, hardware, status, and logs.
- A manifest lists every included file and SHA-256 plus excluded files. Weight
  formats (`.pt`, `.pth`, `.ckpt`, `.safetensors`, `.bin`, `.pkl`, `.pickle`),
  checkpoint directories, extensionless checkpoint files and symlinks are excluded.
  JSON checkpoint metadata is retained.
- With `--push-results`, evidence goes to its own
  `kaggle-results/seed.../<unique session>`-style branch (the actual branch is
  printed in the plan). Source branches are not modified and parallel notebooks
  never push to the same result branch. This branch contains evidence only.
- Push uses `GH_TOKEN` through a temporary Git askpass helper or existing Git
  credentials. The helper and token are not committed. Use `--results-remote`
  to select another configured remote. Omit `--push-results` to export locally.
- If training/evaluation fails, the launcher exports and attempts to push the
  available evidence with an `incomplete` status. It exits with a failure code.
  A failed push also exits with a failure code and leaves the local bundle intact.

`--hours 12 --reserve-minutes 20` is the default. The launcher stops its workers
at the start of the reserve to leave time for exporting and publishing. A Kaggle
VM termination can still prevent that cleanup; save the cloud output and download
checkpoints if you need them for later analysis or recovery. This launcher does
not implement partial-training resume. Specify `--output-root` only with a new,
empty destination; existing output directories are refused.

The measured two-worker timing came from C4 and a short optimizer budget. It is
an estimate, not a guarantee that C1, C3, or C6 will finish before the deadline.
