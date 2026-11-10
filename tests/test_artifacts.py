from __future__ import annotations

import copy

import torch

from artifacts import (
    SeedManifest, atomic_write_json, fingerprint, initialize_lifecycle, manifest_compatibility,
    save_checkpoint, transition_lifecycle, validate_checkpoint, validate_completion, write_manifest,
)
from config import smoke_profile
from controls import construct_model
from recurrence import RecurrenceOptions, backend_metadata


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


def test_evaluation_may_allow_source_revision_drift_only(tmp_path, smoke_config):
    run_dir = tmp_path / "run"
    original = _manifest(smoke_config)
    write_manifest(run_dir / "manifest.json", original)
    revised_source = SeedManifest(
        family=original.family, run_mode=original.run_mode, scale=original.scale, variant=original.variant,
        model=original.model, seeds=original.seeds, budget=original.budget,
        data_identity=original.data_identity, evaluation_protocol=original.evaluation_protocol,
        expected_artifacts=original.expected_artifacts, environment=original.environment,
        source={"revision": "new-evaluation-only-revision", "dirty": False},
    )
    assert not manifest_compatibility(run_dir / "manifest.json", revised_source)[0]
    assert manifest_compatibility(
        run_dir / "manifest.json", revised_source, allow_source_drift=True,
    ) == (True, "compatible")
    changed_budget = SeedManifest(
        family=original.family, run_mode=original.run_mode, scale=original.scale, variant=original.variant,
        model=original.model, seeds=original.seeds, budget={"max_optimizer_steps": 3},
        data_identity=original.data_identity, evaluation_protocol=original.evaluation_protocol,
        expected_artifacts=original.expected_artifacts, environment=original.environment,
        source=revised_source.source,
    )
    assert not manifest_compatibility(
        run_dir / "manifest.json", changed_budget, allow_source_drift=True,
    )[0]


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


def test_recurrence_provenance_drift_and_legacy_gap_are_rejected(tmp_path, smoke_config):
    run_dir = tmp_path / "run"
    qualification = tmp_path / "qualification.json"
    qualification.write_text('{"result":"first"}', encoding="utf-8")
    baseline = _manifest(smoke_config)
    baseline = SeedManifest(
        family=baseline.family, run_mode=baseline.run_mode, scale=baseline.scale, variant=baseline.variant,
        model={**baseline.model, "recurrence": backend_metadata(RecurrenceOptions(chunk_size=64, qualification_artifact=str(qualification)))},
        seeds=baseline.seeds, budget=baseline.budget, data_identity=baseline.data_identity,
        evaluation_protocol=baseline.evaluation_protocol, expected_artifacts=baseline.expected_artifacts,
    )
    raw = write_manifest(run_dir / "manifest.json", baseline)
    artifact_option = {"qualification_artifact": str(qualification)}
    for changed in (
        RecurrenceOptions(backend="cuda-fused", **artifact_option),
        RecurrenceOptions(input_dtype="float32", **artifact_option),
        RecurrenceOptions(state_accumulator_dtype="float32", **artifact_option),
        RecurrenceOptions(chunk_size=32, **artifact_option),
    ):
        expected = SeedManifest(
            family=baseline.family, run_mode=baseline.run_mode, scale=baseline.scale, variant=baseline.variant,
            model={**baseline.model, "recurrence": backend_metadata(changed)}, seeds=baseline.seeds,
            budget=baseline.budget, data_identity=baseline.data_identity,
            evaluation_protocol=baseline.evaluation_protocol, expected_artifacts=baseline.expected_artifacts,
        )
        assert manifest_compatibility(run_dir / "manifest.json", expected) == (
            False, "configuration/environment/source fingerprint differs",
        )
    qualification.write_text('{"result":"second"}', encoding="utf-8")
    changed_qualification = SeedManifest(
        family=baseline.family, run_mode=baseline.run_mode, scale=baseline.scale, variant=baseline.variant,
        model={**baseline.model, "recurrence": backend_metadata(RecurrenceOptions(chunk_size=64, qualification_artifact=str(qualification)))},
        seeds=baseline.seeds, budget=baseline.budget, data_identity=baseline.data_identity,
        evaluation_protocol=baseline.evaluation_protocol, expected_artifacts=baseline.expected_artifacts,
    )
    assert manifest_compatibility(run_dir / "manifest.json", changed_qualification) == (
        False, "configuration/environment/source fingerprint differs",
    )
    del raw["model"]["recurrence"]
    raw["fingerprint"] = fingerprint({key: value for key, value in raw.items() if key != "fingerprint"})
    atomic_write_json(run_dir / "manifest.json", raw)
    assert manifest_compatibility(run_dir / "manifest.json", baseline) == (
        False, "configuration/environment/source fingerprint differs",
    )


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
    revised_source_manifest = SeedManifest(
        family=manifest.family, run_mode=manifest.run_mode, scale=manifest.scale, variant=manifest.variant,
        model=manifest.model, seeds=manifest.seeds, budget=manifest.budget,
        data_identity=manifest.data_identity, evaluation_protocol=manifest.evaluation_protocol,
        expected_artifacts=manifest.expected_artifacts, environment=manifest.environment,
        source={"revision": "later-evaluation-code", "dirty": False},
    )
    assert validate_completion(run_dir, revised_source_manifest, allow_source_drift=True) == (True, "complete")
    corrupt = copy.deepcopy(metrics)
    corrupt["final_validation"]["perplexity"] = float("inf")
    # Atomic writer protects the prior file if serializing invalid JSON fails.
    try:
        atomic_write_json(run_dir / "metrics.json", corrupt)
    except ValueError:
        pass
    assert validate_completion(run_dir, manifest) == (True, "complete")
