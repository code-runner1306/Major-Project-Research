"""Staged research runner for typed-decision fine-tuning and evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextlib import ExitStack
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
import random
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from typing import Any, Iterator, Mapping

import torch

from decision_artifacts import (
    ArtifactError,
    ResumeCheckpointStore,
    atomic_write_json,
    checkpoint_sha256,
    config_fingerprint,
    initialize_lifecycle,
    read_lifecycle,
    transition_lifecycle,
)
from decision_data import (
    DATASET_NAME,
    WORKFLOWS,
    DataConfig as PipelineDataConfig,
    DecisionDataError,
    get_gpt2_encoder,
    load_all_training_workflows,
    load_official_test_workflows,
    prepare_development_data,
    prepare_official_test_data,
    prepare_example_dict,
    write_data_manifest,
)
from decision_evaluation import (
    apply_temperatures,
    fit_primitive_temperatures,
    predict_gdn,
    paired_metric_bootstrap,
    require_test_stage_ready,
    summarize_predictions,
    summarize_latency,
)
from decision_model import GDNDecisionModel
from decision_baseline import run_laya
from decision_research_config import (
    DEFAULT_RESEARCH_CONFIG,
    ResearchConfig,
    TrainingConfig,
)
from decision_training import train_variant
from resource_telemetry import DeviceResourceCapture


DEFAULT_DRIVE_RESULTS_URL = "https://drive.google.com/file/d/1ThcNNaZssxK1lGjWw9830aU1X4G2vHCl/view"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        temporary.replace(path)


def _save_resource_artifact(
    run_dir: Path, key: str, scope: str, measurement: dict[str, Any], *,
    artifact_name: str,
) -> None:
    """Persist optional device-level telemetry without changing research outputs."""
    path = run_dir / "telemetry" / artifact_name
    try:
        existing = _read_json(path) if path.is_file() else {
            "schema_version": 1,
            "note": "Optional whole-device resource telemetry, not process-isolated. Unsupported readings are recorded; energy is never inferred from rated TDP.",
            "measurements": {},
        }
        existing.setdefault("measurements", {})[key] = {
            "scope": scope,
            "telemetry_scope": "whole_gpu_device; may include unrelated device activity",
            **measurement,
        }
        atomic_write_json(path, existing)
    except Exception as exc:  # Optional telemetry must not invalidate a completed evaluation.
        import warnings
        warnings.warn(f"Could not persist optional resource telemetry: {exc}", RuntimeWarning)


def _save_training_resource(run_dir: Path, variant: str, measurement: dict[str, Any]) -> None:
    """Append a fine-tuning session and recompute cumulative per-variant cost."""
    path = run_dir / "telemetry" / "training_resources.json"
    try:
        artifact = _read_json(path) if path.is_file() else {
            "schema_version": 1,
            "note": "Training telemetry covers fine-tuning sessions only, not prior TinyStories pretraining.",
            "sessions": {},
            "totals_by_variant": {},
        }
        started = str(measurement.get("started_utc") or datetime.now(timezone.utc).isoformat())
        phase = str(measurement.get("conditions", {}).get("phase", "production"))
        session_key = f"{variant}.{phase}@{started}"
        artifact.setdefault("sessions", {})[session_key] = {"variant": variant, **measurement}
        sessions = [item for item in artifact["sessions"].values()
                    if item.get("variant") == variant
                    and item.get("conditions", {}).get("phase", "production") == phase]
        wall_values = [float(item["duration_wall_seconds"]) for item in sessions
                       if isinstance(item.get("duration_wall_seconds"), (int, float))]
        end_steps = [int(item.get("conditions", {}).get("optimizer_steps_at_end", 0)) for item in sessions]
        energy_values = [item.get("energy", {}) for item in sessions]
        all_energy_available = bool(energy_values) and all(
            value.get("supported") is True and isinstance(value.get("joules"), (int, float))
            for value in energy_values
        )
        total = {
            "sessions": len(sessions),
            "cumulative_training_wall_seconds": sum(wall_values),
            "completed_optimizer_steps": max(end_steps, default=0),
            "cumulative_gpu_joules": sum(float(value["joules"]) for value in energy_values)
            if all_energy_available else None,
            "cumulative_gpu_energy_supported_for_all_sessions": all_energy_available,
            "pretraining_energy_included": False,
        }
        artifact.setdefault("totals_by_variant_phase", {})[f"{variant}.{phase}"] = total
        if phase == "production":
            artifact.setdefault("totals_by_variant", {})[variant] = total
        atomic_write_json(path, artifact)
    except Exception as exc:  # Optional telemetry must not invalidate a checkpointed training run.
        import warnings
        warnings.warn(f"Could not persist optional training resource telemetry: {exc}", RuntimeWarning)


def _run_tracked_training(
    run_dir: Path, *, variant: str, phase: str, model: GDNDecisionModel,
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]],
    training_config: TrainingConfig, learning_rate: float, seed: int,
    output_dir: Path, run_fingerprint: str,
    resume_store: ResumeCheckpointStore | None = None,
    resume_payload: Mapping[str, Any] | None = None,
    resource_conditions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one train/preflight/tuning session and persist its resource record."""
    device = (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
              else torch.device("cpu"))
    capture = DeviceResourceCapture(device)
    starting_step = int((resume_payload or {}).get("completed_optimizer_step", 0))
    result: dict[str, Any] | None = None
    try:
        with capture:
            result = train_variant(
                model, train_rows, validation_rows, config=training_config,
                backbone_learning_rate=learning_rate, seed=seed,
                output_dir=output_dir, run_fingerprint=run_fingerprint,
                resume_store=resume_store, resume_payload=resume_payload,
            )
    finally:
        record = capture.result or {
            "runtime": {"nvml_available": False, "reason": "measurement did not complete"},
            "energy": {"supported": False, "joules": None, "reason": "measurement did not complete"},
            "activity_summary": {}, "other_gpu_activity": {"status": "unknown"},
            "activity_samples": [],
        }
        if result is not None:
            ending_step = int(result.get("optimizer_steps", starting_step))
        elif resume_store is not None:
            try:
                ending_step = int(resume_store.load_latest(
                    expected_fingerprint=run_fingerprint
                ).payload.get("completed_optimizer_step", starting_step))
            except Exception:
                ending_step = starting_step
        else:
            ending_step = starting_step
        session_steps = max(0, ending_step - starting_step)
        record["scope"] = f"{phase} fine-tuning call, including validation and checkpoint saves; excludes original TinyStories pretraining"
        record["training_elapsed_seconds"] = record.get("duration_wall_seconds")
        record["conditions"] = {
            "phase": phase,
            "variant": variant,
            "precision": training_config.precision,
            "physical_batch_size": training_config.physical_batch_size,
            "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
            "effective_batch_size": training_config.effective_batch_size,
            "max_sequence_length_tokens": int((resource_conditions or {}).get("max_sequence_length_tokens", 0)),
            "train_decisions": len(train_rows),
            "validation_decisions": len(validation_rows),
            "optimizer_steps_at_start": starting_step,
            "optimizer_steps_at_end": ending_step,
            "optimizer_steps_in_session": session_steps,
            "backbone_learning_rate": learning_rate,
            **dict(resource_conditions or {}),
        }
        if result is not None:
            record["trainer_elapsed_seconds"] = result.get("elapsed_seconds")
        energy_record = record.get("energy", {})
        if energy_record.get("supported") and energy_record.get("joules") is not None:
            energy_record["joules_per_optimizer_step"] = (
                energy_record["joules"] / session_steps if session_steps else None
            )
        _save_training_resource(run_dir, variant, record)
    if result is None:
        raise RuntimeError("Training returned no result")
    result["resource_telemetry_artifact"] = "../telemetry/training_resources.json"
    return result


@contextmanager
def _evaluation_energy(
    run_dir: Path, key: str, scope: str, *, conditions: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Record evaluation power, device activity, and runtime context best-effort."""
    device = (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
              else torch.device("cpu"))
    capture = DeviceResourceCapture(device)
    try:
        with capture:
            yield
    finally:
        measurement = capture.result or {
            "runtime": {"nvml_available": False, "reason": "measurement did not complete"},
            "energy": {"supported": False, "joules": None, "reason": "measurement did not complete"},
            "activity_summary": {}, "other_gpu_activity": {"status": "unknown"},
            "activity_samples": [],
        }
        measurement["conditions"] = dict(conditions or {})
        _save_resource_artifact(run_dir, key, scope, measurement,
                                artifact_name="evaluation_energy.json")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _training_config(config: ResearchConfig) -> TrainingConfig:
    return config.training


def _pipeline_data_config(config: ResearchConfig) -> PipelineDataConfig:
    return PipelineDataConfig(
        dataset_name=config.data.dataset_id,
        dataset_revision=config.data.dataset_revision,
        split_seed=config.data.split_seed,
        context_length=config.data.max_length,
        tokenizer_name=config.data.tokenizer_name,
        train_case_count=config.data.train_cases_per_workflow,
        validation_case_count=config.data.validation_cases_per_workflow,
        calibration_case_count=config.data.calibration_cases_per_workflow,
        expected_train_cases=300,
        expected_test_cases=100,
        expected_questions_per_case=config.data.expected_questions_per_case,
    )


def _resolve_dataset_revision(dataset_id: str, revision: str) -> str:
    """Resolve mutable Hub refs to immutable commits before data acquisition."""
    if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision.lower()):
        return revision
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to resolve an immutable dataset revision") from exc
    info = HfApi().dataset_info(dataset_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Could not resolve dataset revision {dataset_id}@{revision} to a commit")
    return str(info.sha)


def _save_serialized_partitions(run_dir: Path, partitions: dict[str, list[Any]]) -> None:
    data_dir = run_dir / "data"
    for partition in ("train", "validation", "calibration"):
        rows = [item.to_dict() for item in partitions[partition]]
        _write_jsonl(data_dir / f"{partition}.jsonl", rows)


def _trainer_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    output = []
    for row in rows:
        provenance = row.get("provenance", {})
        output.append({
            "input_ids": torch.tensor(row["token_ids"], dtype=torch.long),
            "decision_index": int(row["decision_position"]),
            "option_spans": [tuple(map(int, span)) for span in row["option_spans"]],
            "target": torch.tensor(row["target"], dtype=torch.float32),
            "primitive": str(provenance.get("primitive", "")),
            "workflow": str(provenance.get("workflow", "")),
            "case_id": str(provenance.get("case_id", "")),
            "question_id": str(provenance.get("question_id", "")),
            "truncated": bool(provenance.get("state_truncated", False)),
            "original_length": int(provenance.get("state_original_tokens", len(row["token_ids"]))),
            "retained_length": int(provenance.get("state_retained_tokens", len(row["token_ids"]))),
            "canonical_labels": list(provenance.get("canonical_labels", row["labels"])),
            "score_values": row.get("score_values"),
            "state_text": row.get("state_text", provenance.get("retained_state_text", "")),
            "instructions": row.get("instructions", ""),
            "options": row.get("options", []),
            "option_order": list(provenance.get("option_order", range(len(row["labels"])))),
            "target": torch.tensor(row["target"], dtype=torch.float32),
        })
    return output


def prepare_data(config: ResearchConfig, run_dir: Path) -> dict[str, Any]:
    immutable_revision = _resolve_dataset_revision(config.data.dataset_id, config.data.dataset_revision)
    config = replace(config, data=replace(config.data, dataset_revision=immutable_revision))
    data_dir = run_dir / "data"
    existing_split_path = data_dir / "split_manifest.json"
    existing_split = _read_json(existing_split_path) if existing_split_path.exists() else None
    rows = load_all_training_workflows(
        revision=config.data.dataset_revision,
        dataset_name=config.data.dataset_id,
        expected_cases=300,
        expected_questions_per_case=config.data.expected_questions_per_case,
    )
    split_manifest, partitions, audit = prepare_development_data(
        rows,
        config=_pipeline_data_config(config),
        encoder=get_gpt2_encoder(),
        existing_manifest=existing_split,
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(existing_split_path, split_manifest.to_dict())
    write_data_manifest(data_dir / "data_manifest.json", audit)
    _save_serialized_partitions(run_dir, partitions)
    common = {
        "schema_version": 1,
        "dataset": config.data.dataset_id,
        "dataset_revision": config.data.dataset_revision,
        "dataset_split_hash": split_manifest.to_dict()["split_hash"],
        "data_manifest_hash": audit.get("manifest_hash", audit.get("content_hash")),
        "configuration": config.canonical_dict(),
    }
    run_manifest_path = run_dir / "run_manifest.json"
    if run_manifest_path.exists():
        prior = _read_json(run_manifest_path)
        if (prior.get("dataset_split_hash") != common["dataset_split_hash"]
                or prior.get("data_manifest_hash") != common["data_manifest_hash"]):
            raise ArtifactError("Prepared data identity changed; use a new run directory")
        prior.update(common)
        common = prior
    else:
        common["training_config_locked"] = False
    atomic_write_json(run_manifest_path, common)
    return {"split_manifest": split_manifest.to_dict(), "data_manifest": audit, "counts": {
        key: len(value) for key, value in partitions.items()
    }}


@contextmanager
def _checkpoint_paths(args: argparse.Namespace, variant: str) -> Iterator[Path]:
    direct = getattr(args, f"{variant.replace('-', '_')}_checkpoint", None)
    archive = getattr(args, "checkpoint_archive", None)
    archive_url = getattr(args, "checkpoint_archive_url", None)
    if archive is None and archive_url is None:
        if direct is None or not Path(direct).is_file():
            raise FileNotFoundError(
                f"No checkpoint for {variant}. Pass --{variant}-checkpoint or --checkpoint-archive."
            )
        yield Path(direct)
        return
    with tempfile.TemporaryDirectory(prefix="decision_research_ckpt_") as temporary:
        temporary_dir = Path(temporary)
        if archive_url is not None:
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError("Install gdown or mount the archive with --checkpoint-archive") from exc
            archive = temporary_dir / "results.tar.gz"
            downloaded = gdown.download(archive_url, str(archive), fuzzy=True, quiet=False)
            if downloaded is None or not archive.is_file():
                raise RuntimeError("Google Drive checkpoint archive download failed")
        else:
            archive = Path(archive)
        if not archive.is_file():
            raise FileNotFoundError(f"Checkpoint archive not found: {archive}")
        suffix = (
            "tinystories/30M/scalar/seed_43_ef7dde5a783e/final_checkpoint.pt"
            if variant == "scalar" else
            "tinystories/30M/channel-wise/seed_42_ced68e206c8b/final_checkpoint.pt"
        )
        found = None
        with tarfile.open(archive, "r|gz") as members:
            for member in members:
                normalized = member.name.replace("\\", "/").lstrip("./")
                if normalized.endswith(suffix):
                    if found is not None or not member.isfile() or member.size > 1024**3:
                        raise ValueError(f"Invalid or duplicate {variant} checkpoint in archive")
                    source = members.extractfile(member)
                    if source is None:
                        raise ValueError(f"Cannot read {variant} checkpoint member")
                    found = temporary_dir / "pretrained_checkpoint.pt"
                    with source, found.open("wb") as output:
                        shutil.copyfileobj(source, output)
        if found is None:
            raise FileNotFoundError(f"Archive does not contain selected {variant} checkpoint")
        yield found


def _run_fingerprint(config: ResearchConfig, run_dir: Path, variant: str, checkpoint: Path,
                    learning_rate: float) -> str:
    data_manifest = _read_json(run_dir / "data" / "data_manifest.json")
    run_manifest = _read_json(run_dir / "run_manifest.json")
    return config_fingerprint({
        "run_schema": 1,
        "variant": variant,
        "backbone_learning_rate": learning_rate,
        "configuration": config.canonical_dict(),
        "dataset_split_hash": run_manifest.get("dataset_split_hash"),
        "data_manifest": data_manifest,
        "pretrained_checkpoint_sha256": checkpoint_sha256(checkpoint),
        "torch_version": torch.__version__,
    })


def _assert_prepared_profile(config: ResearchConfig, manifest: dict[str, Any]) -> None:
    """Refuse any stage if data/model/training/evaluation settings drift."""
    locked = manifest.get("configuration", {})
    current = config.canonical_dict()
    pinned_revision = str(manifest.get("dataset_revision", ""))
    requested_revision = str(current.get("data", {}).get("dataset_revision", ""))
    if requested_revision not in {"main", pinned_revision}:
        raise ArtifactError("Requested dataset revision differs from the prepared immutable revision")
    current["data"]["dataset_revision"] = pinned_revision
    fields = ("data", "training", "evaluation")
    for field in fields:
        if field not in locked or config_fingerprint(locked[field]) != config_fingerprint(current[field]):
            raise ArtifactError(f"Prepared {field} profile differs from requested configuration")
    locked_model = {key: value for key, value in locked.get("model", {}).items()
                    if key not in {"scalar_checkpoint", "channel_checkpoint"}}
    current_model = {key: value for key, value in current.get("model", {}).items()
                     if key not in {"scalar_checkpoint", "channel_checkpoint"}}
    if config_fingerprint(locked_model) != config_fingerprint(current_model):
        raise ArtifactError("Prepared model architecture differs from requested configuration")


def _assert_locked_profile(config: ResearchConfig, manifest: dict[str, Any]) -> None:
    """Refuse production execution if controlled settings drift after lock."""
    if manifest.get("training_config_locked") is not True:
        raise RuntimeError("Run the lock command before production execution")
    _assert_prepared_profile(config, manifest)
    if config_fingerprint(manifest.get("locked_training_config", {})) != config_fingerprint(asdict(config.training)):
        raise ArtifactError("Training profile differs from immutable locked profile")


def _variant_dir(run_dir: Path, variant: str) -> Path:
    return run_dir / variant


def train_one(
    config: ResearchConfig,
    run_dir: Path,
    variant: str,
    *,
    learning_rate: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    train_rows = _trainer_rows(run_dir / "data" / "train.jsonl")
    validation_rows = _trainer_rows(run_dir / "data" / "validation.jsonl")
    variant_dir = _variant_dir(run_dir, variant)
    checkpoint_dir = variant_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _read_json(run_dir / "run_manifest.json")
    _assert_locked_profile(config, run_manifest)
    if float(run_manifest.get("backbone_learning_rate", -1)) != float(learning_rate):
        raise RuntimeError("Requested learning rate differs from the locked run profile")
    fingerprint = _run_fingerprint(config, run_dir, variant, checkpoint_path, learning_rate)
    lifecycle_path = variant_dir / "lifecycle.json"
    if lifecycle_path.exists():
        lifecycle = read_lifecycle(variant_dir)
        if lifecycle.get("fingerprint") != fingerprint:
            raise ArtifactError("Existing variant run fingerprint differs")
        if lifecycle.get("status") in {"trained", "calibrating", "calibrated", "evaluating", "complete"}:
            prior_result = variant_dir / "training_result.json"
            if prior_result.is_file():
                return _read_json(prior_result)
    else:
        initialize_lifecycle(variant_dir, fingerprint, details={"variant": variant})
    transition_lifecycle(variant_dir, "training", fingerprint)
    model = GDNDecisionModel.from_pretrained_checkpoint(
        checkpoint_path, variant, scale=config.model.scale, projection_dim=config.model.projection_dim
    )
    store = ResumeCheckpointStore(checkpoint_dir, local_root=config.artifacts.local_cache_dir / variant)
    resume_payload = None
    if (checkpoint_dir / "latest_resume.json").exists():
        resume_payload = store.load_latest(expected_fingerprint=fingerprint).payload
    try:
        result = _run_tracked_training(
            run_dir, variant=variant, phase="production", model=model,
            train_rows=train_rows, validation_rows=validation_rows,
            training_config=config.training, learning_rate=learning_rate,
            seed=config.training.fine_tuning_seed, output_dir=variant_dir / "checkpoints",
            run_fingerprint=fingerprint, resume_store=store, resume_payload=resume_payload,
            resource_conditions={"max_sequence_length_tokens": config.data.max_length},
        )
        atomic_write_json(variant_dir / "training_result.json", result)
        transition_lifecycle(variant_dir, "trained", fingerprint, details={
            "optimizer_steps": result["optimizer_steps"], "best_step": result["best_step"]
        })
        return result
    except Exception as exc:
        transition_lifecycle(variant_dir, "failed", fingerprint, failure={
            "stage": "training", "exception_type": type(exc).__name__, "reason": str(exc)
        })
        raise


def tune_learning_rate(config: ResearchConfig, run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Compare the configured learning rates on both variants, then lock one."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run prepare before learning-rate tuning")
    current_manifest = _read_json(manifest_path)
    if current_manifest.get("training_config_locked"):
        raise RuntimeError("Training configuration is already locked")
    _assert_prepared_profile(config, current_manifest)
    train_rows = _trainer_rows(run_dir / "data" / "train.jsonl")
    validation_rows = _trainer_rows(run_dir / "data" / "validation.jsonl")
    candidate_results: list[dict[str, Any]] = []
    tuning_config = replace(
        config.training,
        max_epochs=1,
        checkpoint_every_steps=25,
        validation_every_steps=max(1, math.ceil(
            math.ceil(len(train_rows) / config.training.physical_batch_size)
            / config.training.gradient_accumulation_steps
        )),
    )
    config.artifacts.local_cache_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir = run_dir / "tuning"
    candidates_dir = tuning_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        checkpoints = {
            variant: stack.enter_context(_checkpoint_paths(args, variant))
            for variant in ("scalar", "channel-wise")
        }
        for learning_rate in config.training.candidate_backbone_learning_rates:
            for variant in ("scalar", "channel-wise"):
                candidate_id = f"lr_{learning_rate:.0e}_{variant}"
                candidate_dir = candidates_dir / candidate_id
                candidate_result_path = candidate_dir / "candidate_result.json"
                fingerprint = config_fingerprint({
                    "kind": "learning_rate_development_run",
                    "variant": variant,
                    "learning_rate": learning_rate,
                    "checkpoint_sha256": checkpoint_sha256(checkpoints[variant]),
                    "split_hash": current_manifest["dataset_split_hash"],
                    "training_config": asdict(tuning_config),
                })
                if candidate_result_path.is_file():
                    prior_candidate = _read_json(candidate_result_path)
                    if prior_candidate.get("run_fingerprint") != fingerprint:
                        raise ArtifactError(f"Tuning candidate {candidate_id} fingerprint changed")
                    candidate_results.append(prior_candidate)
                    continue
                candidate_dir.mkdir(parents=True, exist_ok=True)
                model = GDNDecisionModel.from_pretrained_checkpoint(
                    checkpoints[variant], variant, scale=config.model.scale,
                    projection_dim=config.model.projection_dim,
                )
                store = ResumeCheckpointStore(
                    candidate_dir / "resume", local_root=config.artifacts.local_cache_dir / "tuning" / candidate_id
                )
                resume_payload = None
                if (candidate_dir / "resume" / "latest_resume.json").exists():
                    resume_payload = store.load_latest(expected_fingerprint=fingerprint).payload
                result = _run_tracked_training(
                    run_dir, variant=variant,
                    phase=f"learning_rate_tuning_lr_{learning_rate:.0e}",
                    model=model, train_rows=train_rows, validation_rows=validation_rows,
                    training_config=tuning_config, learning_rate=learning_rate,
                    seed=config.training.fine_tuning_seed,
                    output_dir=candidate_dir / "checkpoints", run_fingerprint=fingerprint,
                    resume_store=store, resume_payload=resume_payload,
                    resource_conditions={"max_sequence_length_tokens": config.data.max_length},
                )
                candidate_result = {
                    "variant": variant,
                    "backbone_learning_rate": learning_rate,
                    "validation_soft_nll": result["best_validation_nll"],
                    "optimizer_steps": result["optimizer_steps"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "run_fingerprint": fingerprint,
                }
                atomic_write_json(candidate_result_path, candidate_result)
                candidate_results.append(candidate_result)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    aggregate = {
        rate: sum(row["validation_soft_nll"] for row in candidate_results
                  if row["backbone_learning_rate"] == rate) / 2
        for rate in config.training.candidate_backbone_learning_rates
    }
    selected_rate = min(aggregate, key=lambda rate: (aggregate[rate], rate))
    result = {
        "schema_version": 1,
        "split_hash": current_manifest["dataset_split_hash"],
        "candidate_results": candidate_results,
        "aggregate_validation_soft_nll": {str(key): value for key, value in aggregate.items()},
        "selected_backbone_learning_rate": selected_rate,
        "selection_rule": "lowest mean validation soft NLL across scalar and channel-wise; lower LR breaks ties",
    }
    atomic_write_json(tuning_dir / "learning_rate_results.json", result)
    current_manifest.update({
        "training_config_locked": True,
        "backbone_learning_rate": selected_rate,
        "locked_training_config": asdict(config.training),
        "learning_rate_tuning_sha256": checkpoint_sha256(tuning_dir / "learning_rate_results.json"),
        "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    current_manifest["training_config_hash"] = config_fingerprint({
        "backbone_learning_rate": selected_rate,
        "training": asdict(config.training),
        "data_manifest_hash": current_manifest.get("data_manifest_hash"),
    })
    atomic_write_json(manifest_path, current_manifest)
    return result


def _save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_jsonl(path, rows)


def _canonicalize_prediction_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore permuted vectors and labels to their stable source order."""
    output = []
    for row in rows:
        order = [int(index) for index in row.get("option_order", range(len(row["canonical_labels"])))]
        if sorted(order) != list(range(len(order))):
            raise ValueError("Prediction carries an invalid option-order mapping")
        inverse = [order.index(index) for index in range(len(order))]
        copied = dict(row)
        for key in ("raw_logits", "raw_probabilities", "calibrated_probabilities", "target", "score_values"):
            if row.get(key) is not None:
                copied[key] = [row[key][index] for index in inverse]
        copied["option_order"] = list(range(len(order)))
        output.append(copied)
    return output


def _option_order_robustness(
    model: GDNDecisionModel,
    examples: list[Any],
    canonical_predictions: list[dict[str, Any]],
    temperatures: dict[str, Any],
    *, seed: int,
    permutations: int,
    output_dir: Path,
) -> dict[str, Any]:
    canonical_argmax = [int(max(range(len(row["raw_probabilities"])), key=row["raw_probabilities"].__getitem__))
                        for row in canonical_predictions]
    results = []
    canonical_metrics = summarize_predictions(
        canonical_predictions, probability_key="calibrated_probabilities"
    )["overall"]
    robustness_dir = output_dir / "option_order"
    for permutation_index in range(permutations):
        permuted_rows = []
        for item_index, example in enumerate(examples):
            count = len(example.options)
            order = list(range(count))
            if count > 1:
                random.Random(seed + item_index * 104729 + permutation_index).shuffle(order)
                if order == list(range(count)):
                    order = order[1:] + order[:1]
            permuted_rows.append(prepare_example_dict(example, option_order=order))
        scored = predict_gdn(model, permuted_rows, partition="test")
        restored = _canonicalize_prediction_order(scored)
        calibrated = apply_temperatures(restored, temperatures)
        summary = summarize_predictions(calibrated, probability_key="calibrated_probabilities")
        _save_predictions(robustness_dir / f"permutation_{permutation_index + 1}.jsonl", calibrated)
        permuted_argmax = [int(max(range(len(row["raw_probabilities"])), key=row["raw_probabilities"].__getitem__))
                           for row in restored]
        results.append({
            "permutation": permutation_index,
            "flip_rate": sum(a != b for a, b in zip(canonical_argmax, permuted_argmax)) / len(canonical_argmax),
            "metrics": summary["overall"],
            "accuracy_delta": summary["overall"]["accuracy"] - canonical_metrics["accuracy"],
            "soft_nll_delta": summary["overall"]["soft_nll"] - canonical_metrics["soft_nll"],
        })
    return {"seed": seed, "permutations": results}


def calibrate_variant(config: ResearchConfig, run_dir: Path, variant: str) -> dict[str, Any]:
    variant_dir = _variant_dir(run_dir, variant)
    existing_calibration = variant_dir / "calibration.json"
    if existing_calibration.is_file():
        lifecycle_now = read_lifecycle(variant_dir)
        if lifecycle_now.get("status") in {"calibrated", "evaluating", "complete"}:
            return _read_json(existing_calibration)
    run_manifest = _read_json(run_dir / "run_manifest.json")
    _assert_locked_profile(config, run_manifest)
    lifecycle = read_lifecycle(variant_dir)
    fingerprint = str(lifecycle["fingerprint"])
    transition_lifecycle(variant_dir, "calibrating", fingerprint)
    model = GDNDecisionModel.load_inference_checkpoint(variant_dir / "checkpoints" / "best_model.pt")
    calibration = _trainer_rows(run_dir / "data" / "calibration.jsonl")
    records = predict_gdn(model, calibration, partition="calibration")
    _save_predictions(variant_dir / "calibration_predictions.jsonl", records)
    temperatures = fit_primitive_temperatures(records)
    artifact = {
        "schema_version": 1,
        "variant": variant,
        "run_fingerprint": fingerprint,
        "split_hash": run_manifest["dataset_split_hash"],
        "temperatures": temperatures,
    }
    atomic_write_json(variant_dir / "calibration.json", artifact)
    transition_lifecycle(variant_dir, "calibrated", fingerprint)
    return artifact


def evaluate_variant(config: ResearchConfig, run_dir: Path, variant: str) -> dict[str, Any]:
    variant_dir = _variant_dir(run_dir, variant)
    lifecycle = read_lifecycle(variant_dir)
    completed_metrics = variant_dir / "test_metrics.json"
    if lifecycle.get("status") == "complete" and completed_metrics.is_file():
        return _read_json(completed_metrics)
    run_manifest = _read_json(run_dir / "run_manifest.json")
    _assert_locked_profile(config, run_manifest)
    calibration = _read_json(variant_dir / "calibration.json")
    require_test_stage_ready(manifest=run_manifest, lifecycle=lifecycle, calibration=calibration)
    split_manifest = _read_json(run_dir / "data" / "split_manifest.json")
    pinned_revision = str(run_manifest["dataset_revision"])
    pinned_config = replace(config, data=replace(config.data, dataset_revision=pinned_revision))
    test_rows = load_official_test_workflows(
        revision=pinned_revision,
        dataset_name=config.data.dataset_id,
        expected_cases=100,
        expected_questions_per_case=config.data.expected_questions_per_case,
    )
    test_examples, test_data_manifest = prepare_official_test_data(
        test_rows,
        config=_pipeline_data_config(pinned_config),
        encoder=get_gpt2_encoder(),
        split_manifest=split_manifest,
    )
    rows = [prepare_example_dict(item) for item in test_examples]
    model = GDNDecisionModel.load_inference_checkpoint(variant_dir / "checkpoints" / "best_model.pt")
    evaluation_device = (torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available()
                         else torch.device("cpu"))
    model.to(evaluation_device)
    fingerprint = str(lifecycle["fingerprint"])
    transition_lifecycle(variant_dir, "evaluating", fingerprint)
    warmup_runs = min(config.evaluation.latency_warmups, len(rows))
    if rows:
        warmup_example = rows[0]
        with torch.inference_mode():
            for _ in range(warmup_runs):
                model.score(warmup_example["input_ids"], warmup_example["decision_index"],
                            warmup_example["option_spans"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    evaluation_conditions = {
        "model_variant": variant,
        "weight_dtype": str(next(model.parameters()).dtype),
        "precision": "float32 inference; no autocast",
        "autocast": False,
        "batch_size": 1,
        "max_sequence_length_tokens": config.data.max_length,
        "warmup_runs": warmup_runs,
        "minimum_timed_observations": config.evaluation.latency_repeats,
        "timed_decisions": len(rows),
        "inference_passes_per_decision": 1,
        "input_serialized_length_min": min((len(row["input_ids"]) for row in rows), default=0),
        "input_serialized_length_max": max((len(row["input_ids"]) for row in rows), default=0),
        "option_count_min": min((len(row["canonical_labels"]) for row in rows), default=0),
        "option_count_max": max((len(row["canonical_labels"]) for row in rows), default=0),
    }
    with _evaluation_energy(
        run_dir, f"{variant}.canonical_test_inference",
        "GDN canonical-order official-test predictions; excludes model loading and calibration",
        conditions=evaluation_conditions,
    ):
        predictions = predict_gdn(model, rows, partition="test")
    calibrated = apply_temperatures(predictions, calibration["temperatures"])
    _save_predictions(variant_dir / "test_predictions.jsonl", calibrated)
    permutation_count = max(3, config.evaluation.option_permutations)
    robustness_conditions = dict(evaluation_conditions) | {
        "warmup_runs": 0,
        "warmup_shared_from_canonical_pass": True,
        "timed_decisions": len(rows) * permutation_count,
        "minimum_timed_observations": config.evaluation.latency_repeats,
        "inference_passes_per_decision": permutation_count,
        "permutation_count": permutation_count,
    }
    with _evaluation_energy(
        run_dir, f"{variant}.option_order_robustness_inference",
        "GDN official-test option-order permutations, including per-permutation metric calculations and artifact writes; excludes canonical predictions",
        conditions=robustness_conditions,
    ):
        robustness = _option_order_robustness(
            model, test_examples, calibrated, calibration["temperatures"],
            seed=config.evaluation.bootstrap_seed,
            permutations=permutation_count,
            output_dir=variant_dir,
        )
    permuted_predictions = [
        row
        for permutation_index in range(1, permutation_count + 1)
        for row in _read_jsonl(variant_dir / "option_order" / f"permutation_{permutation_index}.jsonl")
    ]
    robustness["latency_summary"] = summarize_latency(permuted_predictions)
    metrics = {
        "schema_version": 1,
        "variant": variant,
        "run_fingerprint": fingerprint,
        "test_data_manifest": test_data_manifest,
        "raw": summarize_predictions(calibrated, probability_key="raw_probabilities"),
        "calibrated": summarize_predictions(calibrated, probability_key="calibrated_probabilities"),
        "latency": summarize_latency(calibrated),
        "option_order_robustness": robustness,
    }
    atomic_write_json(variant_dir / "test_metrics.json", metrics)
    transition_lifecycle(variant_dir, "complete", fingerprint)
    return metrics


def evaluate_laya(config: ResearchConfig, run_dir: Path) -> dict[str, Any]:
    """Evaluate the optional, directly-executed Laya baseline on official test."""
    run_manifest = _read_json(run_dir / "run_manifest.json")
    _assert_locked_profile(config, run_manifest)
    pinned_revision = str(run_manifest["dataset_revision"])
    pinned_config = replace(config, data=replace(config.data, dataset_revision=pinned_revision))
    split_manifest = _read_json(run_dir / "data" / "split_manifest.json")
    test_rows = load_official_test_workflows(
        revision=pinned_revision, dataset_name=config.data.dataset_id,
        expected_cases=100, expected_questions_per_case=config.data.expected_questions_per_case,
    )
    examples, data_manifest = prepare_official_test_data(
        test_rows, config=_pipeline_data_config(pinned_config), encoder=get_gpt2_encoder(),
        split_manifest=split_manifest,
    )
    rows = [prepare_example_dict(item) for item in examples]
    prior_laya_identity = run_manifest.get("laya_identity", {})
    requested_laya_revision = prior_laya_identity.get("model_revision") or (
        config.baseline.revision if config.baseline.revision != "main" else None
    )
    predictions, baseline_identity = run_laya(rows, model_id=config.baseline.model_id,
                                               revision=requested_laya_revision,
                                               expected_package_version=config.baseline.package_version,
                                               warmup_runs=config.evaluation.latency_warmups)
    laya_resources = baseline_identity.pop("inference_resources", None)
    if laya_resources is not None:
        _save_resource_artifact(
            run_dir, "laya.canonical_test_inference",
            "Laya canonical-order official-test predictions; excludes model loading and warm-up calls",
            laya_resources,
            artifact_name="evaluation_energy.json",
        )
    if prior_laya_identity and prior_laya_identity.get("package_version") != baseline_identity.get("package_version"):
        raise ArtifactError("Installed Laya package version differs from the locked baseline runtime")
    run_manifest["laya_identity"] = baseline_identity
    atomic_write_json(run_dir / "run_manifest.json", run_manifest)
    baseline_dir = run_dir / "laya"
    _save_predictions(baseline_dir / "test_predictions.jsonl", predictions)
    metrics = {
        "schema_version": 1, "variant": "laya", "baseline_identity": baseline_identity,
        "test_data_manifest": data_manifest,
        "native": summarize_predictions(predictions, probability_key="raw_probabilities"),
        "latency": baseline_identity.get("latency_summary", {}),
    }
    atomic_write_json(baseline_dir / "test_metrics.json", metrics)
    return metrics


def generate_report(run_dir: Path) -> dict[str, Any]:
    """Regenerate concise Markdown/CSV tables from immutable metric JSON."""
    variants = ("scalar", "channel-wise", "laya")
    metrics = {}
    for variant in variants:
        path = run_dir / variant / "test_metrics.json"
        if path.is_file():
            metrics[variant] = _read_json(path)
    if len(metrics) < 2:
        raise RuntimeError("Report requires at least two completed official-test metric artifacts")
    rows = []
    breakdown_rows = []
    for variant, artifact in metrics.items():
        section = artifact.get("calibrated", artifact.get("native", {})).get("overall", {})
        rows.append({"model": variant, **section})
        summary = artifact.get("calibrated", artifact.get("native", {}))
        for dimension in ("workflow", "primitive", "option_count", "truncated"):
            for group, values in summary.get(dimension, {}).items():
                breakdown_rows.append({"model": variant, "dimension": dimension,
                                       "group": group, **values})
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with (report_dir / "overall_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    markdown = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    markdown.extend("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows)
    (report_dir / "overall_metrics.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    breakdown_columns = ["model", "dimension", "group", *[key for key in rows[0] if key != "model"]]
    with (report_dir / "breakdown_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=breakdown_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(breakdown_rows)
    latency_rows: list[dict[str, Any]] = []
    latency_artifact: dict[str, Any] = {}
    for model_name, artifact in metrics.items():
        model_latency: dict[str, Any] = {"canonical": artifact.get("latency", {})}
        if "option_order_robustness" in artifact:
            model_latency["option_order_robustness"] = artifact["option_order_robustness"].get("latency_summary", {})
        latency_artifact[model_name] = model_latency
        for protocol, latency_summary in model_latency.items():
            for dimension in (
                "overall", "primitive", "serialized_length_bucket", "option_count",
                "primitive_x_length_bucket",
            ):
                values = latency_summary.get(dimension, {})
                if dimension == "overall" and values:
                    latency_rows.append({"model": model_name, "protocol": protocol,
                                         "dimension": dimension, "group": "all",
                                         "input_length_unit": latency_summary.get("input_length_unit"), **values})
                elif isinstance(values, dict):
                    for group, stats in values.items():
                        latency_rows.append({"model": model_name, "protocol": protocol,
                                             "dimension": dimension, "group": group,
                                             "input_length_unit": latency_summary.get("input_length_unit"), **stats})
    atomic_write_json(report_dir / "latency_summary.json", latency_artifact)
    if latency_rows:
        latency_columns = ["model", "protocol", "dimension", "group", "input_length_unit", "decisions",
                           "mean_seconds", "median_seconds", "p95_seconds", "decisions_per_second"]
        with (report_dir / "latency_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=latency_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(latency_rows)
    paired = {}
    prediction_paths = {name: run_dir / name / "test_predictions.jsonl" for name in metrics}
    for left, right in (("scalar", "channel-wise"), ("scalar", "laya"), ("channel-wise", "laya")):
        if left in metrics and right in metrics and prediction_paths[left].is_file() and prediction_paths[right].is_file():
            paired[f"{left}_minus_{right}"] = paired_metric_bootstrap(
                _read_jsonl(prediction_paths[left]), _read_jsonl(prediction_paths[right]),
                probability_key="calibrated_probabilities",
                replicates=DEFAULT_RESEARCH_CONFIG.evaluation.bootstrap_replicates,
                seed=DEFAULT_RESEARCH_CONFIG.evaluation.bootstrap_seed,
            )
    atomic_write_json(report_dir / "paired_case_bootstrap.json", paired)
    reliability_rows = []
    for model_name in metrics:
        prediction_path = run_dir / model_name / "test_predictions.jsonl"
        if not prediction_path.is_file():
            continue
        for prediction in _read_jsonl(prediction_path):
            probs = prediction.get("calibrated_probabilities", prediction["raw_probabilities"])
            chosen = max(range(len(probs)), key=probs.__getitem__)
            reliability_rows.append((model_name, float(probs[chosen]),
                                     float(prediction["target"][chosen])))
    reliability_bins = []
    for model_name in metrics:
        model_rows = [row for row in reliability_rows if row[0] == model_name]
        for bin_index in range(10):
            members = [row for row in model_rows if min(int(row[1] * 10), 9) == bin_index]
            if members:
                reliability_bins.append({
                    "model": model_name, "bin": bin_index,
                    "mean_confidence": sum(row[1] for row in members) / len(members),
                    "soft_accuracy": sum(row[2] for row in members) / len(members),
                    "count": len(members),
                })
    with (report_dir / "reliability_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "bin", "mean_confidence", "soft_accuracy", "count"])
        writer.writeheader()
        writer.writerows(reliability_bins)
    try:
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.7,
                             "pdf.fonttype": 42, "ps.fonttype": 42})
        figure, axis = plt.subplots(figsize=(4.2, 3.2), constrained_layout=True)
        axis.plot([0, 1], [0, 1], color="0.55", linewidth=0.8, linestyle="--", label="ideal")
        markers = ("o", "s", "^", "D")
        for index, model_name in enumerate(metrics):
            points = [row for row in reliability_bins if row["model"] == model_name]
            axis.plot([row["mean_confidence"] for row in points],
                      [row["soft_accuracy"] for row in points], marker=markers[index % len(markers)],
                      markersize=3.5, linewidth=1.1, label=model_name)
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
                 ylabel="Soft correctness", title="Reliability on official test set")
        axis.grid(True, color="0.9", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)
        figure.savefig(report_dir / "reliability_diagram.pdf")
        figure.savefig(report_dir / "reliability_diagram.png", dpi=300)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
        x = list(range(len(rows)))
        width = 0.36
        axis.bar([value - width / 2 for value in x], [row.get("accuracy", 0) for row in rows],
                 width, color="0.35", label="Accuracy")
        axis.bar([value + width / 2 for value in x], [row.get("soft_accuracy", 0) for row in rows],
                 width, color="0.72", label="Soft accuracy")
        axis.set_xticks(x, [row["model"] for row in rows], rotation=15, ha="right")
        axis.set_ylim(0, 1)
        axis.set_ylabel("Score")
        axis.set_title("Decision quality")
        axis.grid(axis="y", color="0.9", linewidth=0.5)
        axis.legend(frameon=False, fontsize=8)
        figure.savefig(report_dir / "quality_comparison.pdf")
        figure.savefig(report_dir / "quality_comparison.png", dpi=300)
        plt.close(figure)

        robustness_rows = []
        for model_name, artifact in metrics.items():
            for result in artifact.get("option_order_robustness", {}).get("permutations", []):
                robustness_rows.append({"model": model_name, **result})
        if robustness_rows:
            with (report_dir / "option_order_robustness.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["model", "permutation", "flip_rate",
                                                           "accuracy_delta", "soft_nll_delta"])
                writer.writeheader()
                writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in robustness_rows)
            figure, axis = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
            for index, model_name in enumerate(metrics):
                points = [row for row in robustness_rows if row["model"] == model_name]
                axis.plot([row["permutation"] + 1 for row in points], [row["flip_rate"] for row in points],
                          marker=markers[index % len(markers)], linewidth=1.1, label=model_name)
            axis.set(xlabel="Deterministic option permutation", ylabel="Prediction flip rate",
                     ylim=(0, 1), title="Option-order robustness")
            axis.grid(True, color="0.9", linewidth=0.5)
            axis.legend(frameon=False, fontsize=8)
            figure.savefig(report_dir / "option_order_robustness.pdf")
            figure.savefig(report_dir / "option_order_robustness.png", dpi=300)
            plt.close(figure)
    except ImportError:
        pass
    summary_text = """# Decision fine-tuning research summary

This report compares the selected scalar and channel-wise 30M Gated DeltaNet checkpoints after decision fine-tuning with direct Laya inference. Metrics are derived from saved per-question predictions; reported intervals resample cases to retain within-case dependence.

## Interpretation limits

- The two Gated DeltaNet variants originate from different pretraining seeds (scalar seed 43; channel-wise seed 42), so differences cannot be attributed solely to decay granularity.
- The language-model checkpoints were selected from the available prior runs; this is not a multi-seed pretraining comparison.
- Decision targets are the dataset's supplied soft labels; findings describe agreement with that teacher distribution and do not establish real-world decision validity.
- Truncated states are represented by the recorded head-and-tail policy; results do not establish performance on arbitrarily long contexts.
- Laya is a public typed-decision model evaluated directly. Its training exposure to this benchmark family must be disclosed and checked before interpreting it as an independent held-out competitor.
- Laya has approximately 421M parameters versus approximately 30M for each Gated DeltaNet, so this is not a parameter-matched comparison.

See the machine-readable metric, grouped latency, bootstrap, prediction, training-resource, and GPU-device telemetry artifacts alongside this summary. Fine-tuning cost includes resumed sessions recorded by this pipeline and excludes the original TinyStories pretraining.
"""
    (report_dir / "research_summary.md").write_text(summary_text, encoding="utf-8")
    return {"models": list(metrics), "report_dir": str(report_dir), "rows": rows,
            "paired_comparisons": list(paired)}


def package_results(run_dir: Path) -> dict[str, Any]:
    """Package results only after core model/test/calibration artifacts exist."""
    required = [
        run_dir / "scalar" / "checkpoints" / "best_model.pt",
        run_dir / "channel-wise" / "checkpoints" / "best_model.pt",
        run_dir / "scalar" / "calibration.json", run_dir / "channel-wise" / "calibration.json",
        run_dir / "scalar" / "test_predictions.jsonl", run_dir / "channel-wise" / "test_predictions.jsonl",
        run_dir / "laya" / "test_predictions.jsonl", run_dir / "reports" / "overall_metrics.csv",
    ]
    missing = [str(path.relative_to(run_dir)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Cannot package an incomplete study; missing: " + ", ".join(missing))
    import hashlib
    import tarfile
    archive_path = run_dir / config_archive_name(run_dir)
    inventory_path = run_dir / "artifact_inventory.json"
    candidates = [path for path in sorted(run_dir.rglob("*")) if path.is_file()
                  and path not in {archive_path, inventory_path}
                  and path.name not in {"resume_A.pt", "resume_B.pt", "latest_resume.json", "final_checkpoint.pt"}
                  and "pretrained_checkpoint" not in path.parts
                  and not ("tuning" in path.parts and "candidates" in path.parts and "checkpoints" in path.parts)]
    inventory = [{"path": str(path.relative_to(run_dir)), "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in candidates]
    atomic_write_json(inventory_path, {"schema_version": 1, "artifacts": inventory,
                                       "inventory_excluded_from_own_hashes": True})
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in candidates + [inventory_path]:
            archive.add(path, arcname=str(path.relative_to(run_dir)))
    return {"archive": str(archive_path), "artifacts": len(inventory)}


def config_archive_name(run_dir: Path) -> str:
    return "decision_research_results.tar.gz"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RESEARCH_CONFIG.output_dir)
    parser.add_argument("--dataset-revision", default=DEFAULT_RESEARCH_CONFIG.data.dataset_revision)
    parser.add_argument("--max-length", type=int, default=DEFAULT_RESEARCH_CONFIG.data.max_length)
    parser.add_argument("--physical-batch-size", type=int, default=DEFAULT_RESEARCH_CONFIG.training.physical_batch_size)
    parser.add_argument("--gradient-accumulation", type=int, default=DEFAULT_RESEARCH_CONFIG.training.gradient_accumulation_steps)
    parser.add_argument("--epochs", type=int, default=DEFAULT_RESEARCH_CONFIG.training.max_epochs)
    parser.add_argument("--output-root-local", type=Path, default=DEFAULT_RESEARCH_CONFIG.artifacts.local_cache_dir)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="prepare deterministic train/validation/calibration data")
    lock_parser = subparsers.add_parser("lock", help="freeze the selected common training profile")
    lock_parser.add_argument("--learning-rate", type=float, required=True)
    tune_parser = subparsers.add_parser("tune", help="compare candidate learning rates on validation data")
    tune_parser.add_argument("--checkpoint", type=Path)
    tune_parser.add_argument("--scalar-checkpoint", type=Path)
    tune_parser.add_argument("--channel-wise-checkpoint", type=Path)
    tune_parser.add_argument("--checkpoint-archive", type=Path)
    tune_parser.add_argument("--checkpoint-archive-url", nargs="?", const=DEFAULT_DRIVE_RESULTS_URL)
    preflight_parser = subparsers.add_parser("preflight", help="run a short GPU fit/resume preflight")
    preflight_parser.add_argument("--variant", choices=("scalar", "channel-wise"), default="scalar")
    preflight_parser.add_argument("--checkpoint", type=Path)
    preflight_parser.add_argument("--checkpoint-archive", type=Path)
    train_parser = subparsers.add_parser("train", help="train or resume one selected GDN variant")
    train_parser.add_argument("--variant", choices=("scalar", "channel-wise"), required=True)
    train_parser.add_argument("--learning-rate", type=float, required=True)
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument("--checkpoint-archive", type=Path)
    train_parser.add_argument("--checkpoint-archive-url", nargs="?", const=DEFAULT_DRIVE_RESULTS_URL)
    calibration_parser = subparsers.add_parser("calibrate", help="fit primitive temperatures on calibration split")
    calibration_parser.add_argument("--variant", choices=("scalar", "channel-wise"), required=True)
    evaluation_parser = subparsers.add_parser("evaluate", help="run the locked official test evaluation")
    evaluation_parser.add_argument("--variant", choices=("scalar", "channel-wise"), required=True)
    subparsers.add_parser("baseline", help="run optional direct Laya official-test evaluation")
    subparsers.add_parser("report", help="regenerate tables from completed metric artifacts")
    subparsers.add_parser("package", help="validate and archive study outputs")
    return parser


def _make_config(args: argparse.Namespace) -> ResearchConfig:
    return replace(
        DEFAULT_RESEARCH_CONFIG,
        output_dir=args.run_dir,
        data=replace(DEFAULT_RESEARCH_CONFIG.data, dataset_revision=args.dataset_revision, max_length=args.max_length),
        training=replace(
            DEFAULT_RESEARCH_CONFIG.training,
            physical_batch_size=args.physical_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            max_epochs=args.epochs,
        ),
        artifacts=replace(DEFAULT_RESEARCH_CONFIG.artifacts, output_dir=args.run_dir,
                          local_cache_dir=args.output_root_local),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _make_config(args)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        result = prepare_data(config, run_dir)
    elif args.command == "lock":
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Run prepare before locking the training configuration")
        if args.learning_rate not in config.training.candidate_backbone_learning_rates:
            parser.error("learning rate must be one of the configured candidate values")
        manifest = _read_json(manifest_path)
        _assert_prepared_profile(config, manifest)
        configured_training = manifest.get("configuration", {}).get("training", {})
        if config_fingerprint(configured_training) != config_fingerprint(asdict(config.training)):
            raise ArtifactError("Requested training settings differ from the prepared run configuration")
        manifest.update({
            "training_config_locked": True,
            "backbone_learning_rate": args.learning_rate,
            "locked_training_config": asdict(config.training),
            "locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        manifest["training_config_hash"] = config_fingerprint({
            "backbone_learning_rate": args.learning_rate,
            "training": asdict(config.training),
            "data_manifest_hash": manifest.get("data_manifest_hash"),
        })
        atomic_write_json(manifest_path, manifest)
        result = {"training_config_locked": True, "backbone_learning_rate": args.learning_rate,
                  "training_config_hash": manifest["training_config_hash"]}
    elif args.command == "tune":
        # A shared --checkpoint is allowed for development only when both
        # variants were exported separately elsewhere; named inputs are safer.
        if args.checkpoint is not None:
            parser.error("Use --scalar-checkpoint and --channel-wise-checkpoint, or --checkpoint-archive")
        if args.scalar_checkpoint is not None:
            config = replace(config, model=replace(config.model, scalar_checkpoint=args.scalar_checkpoint))
        if args.channel_wise_checkpoint is not None:
            config = replace(config, model=replace(config.model, channel_checkpoint=args.channel_wise_checkpoint))
        result = tune_learning_rate(config, run_dir, args)
    elif args.command in {"train", "preflight"}:
        variant = args.variant
        checkpoint = args.checkpoint
        if checkpoint is None:
            default_path = config.model.checkpoint_for(variant)
            checkpoint = default_path if default_path.is_file() else None
        if checkpoint is None and args.checkpoint_archive is None and getattr(args, "checkpoint_archive_url", None) is None:
            parser.error("Pass --checkpoint or --checkpoint-archive (or mount checkpoints at configured paths)")
        if not (run_dir / "data" / "train.jsonl").is_file():
            raise FileNotFoundError("Prepared data is missing; run the prepare command first")
        with (_checkpoint_paths(args, variant) if checkpoint is None else _direct_checkpoint(checkpoint)) as checkpoint_path:
            if args.command == "preflight":
                train_rows = _trainer_rows(run_dir / "data" / "train.jsonl")[:32]
                validation_rows = _trainer_rows(run_dir / "data" / "validation.jsonl")[:16]
                small = replace(config.training, max_epochs=1, checkpoint_every_steps=1,
                                validation_every_steps=1, physical_batch_size=min(2, config.training.physical_batch_size),
                                gradient_accumulation_steps=1)
                model = GDNDecisionModel.from_pretrained_checkpoint(checkpoint_path, variant)
                result = _run_tracked_training(
                    run_dir, variant=variant, phase="gpu_preflight", model=model,
                    train_rows=train_rows, validation_rows=validation_rows,
                    training_config=small, learning_rate=2e-5,
                    seed=config.training.fine_tuning_seed,
                    output_dir=run_dir / "preflight" / variant,
                    run_fingerprint=config_fingerprint({"preflight": True, "variant": variant,
                                                       "checkpoint_sha256": checkpoint_sha256(checkpoint_path)}),
                    resource_conditions={"max_sequence_length_tokens": config.data.max_length},
                )
            else:
                result = train_one(config, run_dir, variant,
                                   learning_rate=args.learning_rate, checkpoint_path=checkpoint_path)
    elif args.command == "calibrate":
        result = calibrate_variant(config, run_dir, args.variant)
    elif args.command == "evaluate":
        result = evaluate_variant(config, run_dir, args.variant)
    elif args.command == "baseline":
        result = evaluate_laya(config, run_dir)
    elif args.command == "report":
        result = generate_report(run_dir)
    else:
        result = package_results(run_dir)
    print(json.dumps(result, indent=2, default=str))
    return 0


@contextmanager
def _direct_checkpoint(path: Path) -> Iterator[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    yield path


if __name__ == "__main__":
    raise SystemExit(main())
