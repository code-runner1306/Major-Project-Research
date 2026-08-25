from __future__ import annotations

import copy

import torch

from artifacts import (
    SeedManifest, atomic_write_json, initialize_lifecycle, manifest_compatibility,
    save_checkpoint, transition_lifecycle, validate_checkpoint, validate_completion, write_manifest,
)
from config import smoke_profile
from controls import construct_model


def _manifest(profile, *, run_mode="smoke"):
    return SeedManifest(
        family="tinystories", run_mode=run_mode, scale="tiny", variant="scalar",
        model=profile.model_for("tiny").to_dict(), seeds={"research_seed": 7, "model_seed": 7, "data_seed": 8, "evaluation_seed": 9},
        budget={"max_optimizer_steps": 2}, data_identity={"id": "toy"},
        evaluation_protocol={"version": 1}, expected_artifacts=("manifest.json", "lifecycle.json", "metrics.json", "final_checkpoint.pt"),
    )


def test_smoke_manifest_cannot_resume_as_production(tmp_path, smoke_config):
    run_dir = tmp_path / "run"
    smoke = _manifest(smoke_config)
    write_manifest(run_dir / "manifest.json", smoke)
    production = _manifest(smoke_config, run_mode="production")
    compatible, reason = manifest_compatibility(run_dir / "manifest.json", production)
    assert not compatible and "run mode" in reason


def test_configuration_drift_and_corrupt_checkpoint_are_rejected(tmp_path, smoke_config):
    run_dir = tmp_path / "run"
    manifest = _manifest(smoke_config)
    write_manifest(run_dir / "manifest.json", manifest)
    drifted = SeedManifest(
        family=manifest.family, run_mode=manifest.run_mode, scale=manifest.scale, variant=manifest.variant,
        model=manifest.model, seeds=manifest.seeds, budget={"max_optimizer_steps": 3},
        data_identity=manifest.data_identity, evaluation_protocol=manifest.evaluation_protocol,
        expected_artifacts=manifest.expected_artifacts,
    )
    assert manifest_compatibility(run_dir / "manifest.json", drifted) == (False, "configuration/environment/source fingerprint differs")
    corrupt = run_dir / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    valid, reason, _ = validate_checkpoint(corrupt, 2, 4)
    assert not valid and "cannot be loaded" in reason


def test_atomic_lifecycle_checkpoint_and_completeness(tmp_path, smoke_config):
    run_dir = tmp_path / "run"
    manifest = _manifest(smoke_config)
    data = write_manifest(run_dir / "manifest.json", manifest)
    initialize_lifecycle(run_dir, data["fingerprint"])
    transition_lifecycle(run_dir, "training", data["fingerprint"])
    model, _, _ = construct_model(smoke_config.model_for("tiny"), "scalar", smoke_config.vocab_size)
    checkpoint = save_checkpoint(run_dir / "final_checkpoint.pt", model=model, optimizer=None, completed_steps=2, processed_tokens=4)
    transition_lifecycle(run_dir, "trained", data["fingerprint"])
    transition_lifecycle(run_dir, "evaluating", data["fingerprint"])
    metrics = {
        "training": {"processed_tokens": 4}, "final_checkpoint": checkpoint,
        "final_validation": {"perplexity": 1.0}, "benchmarks": {}, "energy": {"supported": False},
        "gate_diagnostics": {}, "static_cost": {},
    }
    atomic_write_json(run_dir / "metrics.json", metrics)
    transition_lifecycle(run_dir, "complete", data["fingerprint"])
    assert validate_completion(run_dir, manifest) == (True, "complete")
    corrupt = copy.deepcopy(metrics)
    corrupt["final_validation"]["perplexity"] = float("inf")
    # Atomic writer protects the prior file if serializing invalid JSON fails.
    try:
        atomic_write_json(run_dir / "metrics.json", corrupt)
    except ValueError:
        pass
    assert validate_completion(run_dir, manifest) == (True, "complete")
