## ADDED Requirements

### Requirement: Per-seed deterministic dataset shuffling
The `TinyStoriesDataset` loader SHALL accept a random `seed` parameter and apply deterministic dataset shuffling using `.shuffle(seed=seed, buffer_size=10000)` on the streaming dataset.

#### Scenario: Instantiating dataloader with specific seed
- **WHEN** `get_dataloader` is called with `seed=42`
- **THEN** the dataset loader applies a deterministic random shuffle sequence generated specifically from seed 42.

### Requirement: Cross-model dataset split consistency
All model scales (10M, 30M, 55M, 100M) and gating variants (Scalar Gating vs Channel-wise Gating) assigned to a given random seed $S$ SHALL be trained on the exact same dataset split and token sequence generated for seed $S$.

#### Scenario: Training multiple variants under the same seed
- **WHEN** training 10M Variant A with `seed=42` and 100M Variant B with `seed=42`
- **THEN** both models receive identical token stream batches in the exact same order for seed 42.

### Requirement: Per-seed split manifest persistence and resumption
The system SHALL save a per-seed dataset split manifest `data/splits/seed_<seed>_split.json` containing the split random seed, buffer parameters, and dataset metadata. Upon resuming a run, `get_dataloader` SHALL read the existing split manifest for that seed to ensure identical data stream initialization.

#### Scenario: Resuming an interrupted run with saved data split
- **WHEN** `run_all.py` resumes execution for a seed configuration whose split manifest `data/splits/seed_42_split.json` exists
- **THEN** the dataset loader reads the saved split parameters from disk and initializes the exact same data stream.
