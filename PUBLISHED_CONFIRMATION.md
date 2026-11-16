# Published checkpoint confirmation

Run from the repository root with the existing Python environment:

```powershell
.\run_published_confirmation.ps1 -Python .\.venv\Scripts\python.exe
```

The launcher automatically uses CUDA if available, otherwise CPU. To select it
explicitly, add `-Device cpu` or `-Device cuda`. Only one model is loaded onto the
GPU at a time, so one GPU suffices. No optimizer steps or training occur.

The four final checkpoints live in `published_mqar/{scalar,channel-wise}` and
`published_overwrite/{scalar,channel_wise}`. The underscore spelling is accepted
for the downloaded overwrite folder. If a checkpoint has been unpacked as a
PyTorch archive, the helper rebuilds `latest_checkpoint.pt` without deleting or
changing those unpacked files. Only these four `.pt` locations are exceptions in
`.gitignore`; other checkpoint files remain ignored. The unpacked archive folders
are ignored as well. Repacked files have a new archive hash but unchanged stored
tensors/settings; archive reassembly is not training or checkpoint conversion.

Preparation verification without confirmation evaluation:

```powershell
.\run_published_confirmation.ps1 -Python .\.venv\Scripts\python.exe -PrepareOnly
```

The confirmation protocol is written before any scores: 512 examples per task,
seed300042 for training seed42, same task shapes, answer-free primary evaluation
and teacher-forced secondary evaluation, final epoch64 checkpoints for every
condition. The evaluator checks the checkpoint source identity (allowing Windows
newline conversion), final epoch/step counters, strict model weight loading,
paired settings, and actual confirmation overlap with regenerated train and
development splits. It records dataset and checkpoint hashes. The seed and size
are fixed in the evaluator rather than adjustable after seeing results.

Reports are saved under `research_results/published_confirmation_<timestamp>/`.
`confirmation_results.json` includes accuracy, latest-value accuracy, unrelated
retention, stale errors, and evaluation time. The program preserves a failed
status on errors. Treat this as same-distribution confirmation for one seed,
not proof of harder-task generalization or statistical gate superiority. Once
examined, this confirmation set must not be used for tuning a new recipe.
