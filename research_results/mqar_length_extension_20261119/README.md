# Frozen MQAR length extrapolation

The sweep starts from the 12 frozen MQAR checkpoints used in fresh confirmation. They were trained at length 64. The same four associations and query sequence are evaluated at lengths 64, 256, and 1,024. The scripts insert repeated token ID 0 after the initial eight write tokens. This is the filler ID used by the training generator. See protocol.json and checkpoint_hash_audit.json for seeds and checkpoint provenance.

From the repository root, run python replay_mqar_targeted.py to reproduce the two requested seed-44 cells. The 256-token default channel-wise cell produces 301/1,024 correct queries. The 1,024-token mean-decay scalar cell produces 0/1,024. The archived NPZ files record input tokens, labels, query positions, predictions, repeated predictions, top-two logits, and top-1 probabilities. The aggregate sweep covers 36 cells. Per-query archives in this directory cover the two targeted replays. The optional --full-sweep flag produces per-query files for all 36 cells in a separate output folder.

Run python run_paper_filler_decay_diagnostic.py --output NEW_RESULTS_DIR --figure NEW_FIGURE.pdf to rerun the exploratory gate probe. Its saved protocol and summaries are in ../filler_decay_diagnostic_20261119. Existing model weights are required in the checkpoint directories listed by the audit.

Run python run_paper_mqar_length_extension.py --output NEW_RESULTS_DIR to regenerate the 36 aggregate measurements.
