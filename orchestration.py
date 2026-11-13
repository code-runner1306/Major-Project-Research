"""Shared strict orchestration for TinyStories, MQAR, and overwrite recall."""

from __future__ import annotations

from dataclasses import replace
from itertools import islice
import json
from pathlib import Path
import time
from typing import Any, Literal

import torch

from artifacts import (
    LEGACY_PROTOCOL_VERSION, SeedManifest, atomic_write_json, fingerprint, initialize_lifecycle, manifest_compatibility,
    read_json, transition_lifecycle, validate_completion, validate_checkpoint, write_manifest,
)
from config import ExperimentConfig, ModelConfig, RetrievalConfig
from controls import construct_model
from dataset import tinystories_dataloader, toy_dataloader
from diagnostics import collect_gate_diagnostics, static_cost_report
from energy import EnergyProvider
from eval import benchmark_inference, evaluate_needle_suite, generation_cache_equivalence, persisted_needle_suite
from preflight import require_production_cuda
from recurrence import RecurrenceOptions, backend_metadata, ensure_backend_available, qualification_hash
from reporting import write_first_submission_reports
from retrieval import GENERATOR_VERSION, RetrievalDataset, evaluate_retrieval, retrieval_dataloader
from resource_telemetry import DeviceResourceCapture
from train import SeedBundle, evaluate_model, seed_everything, train_fixed_budget


Family = Literal["tinystories", "mqar", "overwrite"]


def recurrence_options_for(profile: ExperimentConfig) -> RecurrenceOptions:
    return RecurrenceOptions(
        backend=profile.recurrence_backend,
        input_dtype=profile.recurrence_input_dtype,
        state_accumulator_dtype=profile.state_accumulator_dtype,
        chunk_size=profile.recurrence_chunk_size,
        qualification_artifact=(
            str(profile.recurrence_qualification_artifact)
            if profile.recurrence_qualification_artifact is not None else None
        ),
    )


def _run_path(profile: ExperimentConfig, family: Family, scale: str, variant: str, seed: int) -> Path:
    if profile.study_task_run_dir is not None:
        return profile.study_task_run_dir
    identity = {
        "run_mode": profile.run_mode, "family": family, "scale": scale, "variant": variant,
        "profile": profile.to_manifest_dict(), "seed": seed,
    }
    if profile.experiment_id is not None:
        identity["study"] = {
            "experiment_id": profile.experiment_id,
            "condition_id": profile.condition_id,
            "initialization": profile.initialization,
        }
    return profile.output_dir / family / scale / variant / f"seed_{seed}_{fingerprint(identity)[:12]}"


def _resolved_backbone(profile: ExperimentConfig, family: Family, scale: str) -> ModelConfig:
    backbone = profile.model_for(scale)
    return replace(backbone, attention_layout="pure_gdn") if family in {"mqar", "overwrite"} else backbone


def _data_for_run(
    profile: ExperimentConfig, family: Family, seeds: SeedBundle, run_dir: Path
) -> tuple[Any, Any, dict[str, Any], Any | None, Any | None, list[tuple[int, Any]]]:
    if family == "tinystories":
        if profile.run_mode == "smoke":
            data_identity = {
                "dataset": "toy-language-v1", "data_seed": seeds.data_seed,
                "train_examples": 12, "validation_examples": 4, "paired_order": "index_order",
            }
            if profile.curriculum:
                raise ValueError("Curriculum is supported only for synthetic retrieval workers")
            return toy_dataloader(profile, seed=seeds.data_seed, split="train"), toy_dataloader(profile, seed=seeds.data_seed, split="validation"), data_identity, None, None, []
        train = tinystories_dataloader(profile, split="train", data_seed=seeds.data_seed, run_dir=run_dir)
        validation = tinystories_dataloader(profile, split="validation", data_seed=seeds.data_seed, run_dir=run_dir)
        data_identity = read_json(run_dir / "data_identity.json")
        data_identity["paired_order"] = "datasets.Dataset.shuffle(seed=data_seed)"
        if profile.curriculum:
            raise ValueError("Curriculum is supported only for synthetic retrieval workers")
        return train, validation, data_identity, None, None, []

    task = "mqar" if family == "mqar" else "overwrite"
    train_set = RetrievalDataset(
        task=task, examples=profile.retrieval.train_examples, config=profile.retrieval,
        seed=seeds.data_seed, split="train",
    )
    study = profile.experiment_id is not None
    validation_seed = seeds.evaluation_seed if study else seeds.evaluation_seed
    test_seed = seeds.evaluation_seed + 1 if study else seeds.evaluation_seed
    validation_retrieval = RetrievalDataset(
        task=task, examples=profile.retrieval.validation_examples or profile.retrieval.held_out_examples, config=profile.retrieval,
        seed=validation_seed, split="validation" if study else "held_out",
    )
    held_out = RetrievalDataset(
        task=task, examples=profile.retrieval.held_out_examples, config=profile.retrieval,
        seed=test_seed, split="held_out",
    )
    train_manifest, validation_manifest, held_manifest = train_set.manifest(), validation_retrieval.manifest(), held_out.manifest()
    if train_manifest["hash"] == held_manifest["hash"]:
        raise AssertionError("Retrieval train and held-out generator identities must not overlap")
    if study and (train_manifest["hash"] == validation_manifest["hash"] or validation_manifest["hash"] == held_manifest["hash"]):
        raise AssertionError("Study retrieval train, validation, and held-out identities must not overlap")
    atomic_write_json(run_dir / "generator_train.json", train_manifest)
    atomic_write_json(run_dir / "generator_held_out.json", held_manifest)
    if study:
        atomic_write_json(run_dir / "generator_validation.json", validation_manifest)
    identity = {
        "generator_version": train_manifest["generator_version"], "train_hash": train_manifest["hash"],
        "held_out_hash": held_manifest["hash"], "paired_order": "indexed_seeded_order",
    }
    if study:
        identity.update({"validation_hash": validation_manifest["hash"], "validation_seed": validation_seed, "test_seed": test_seed})
    curriculum_loaders: list[tuple[int, Any]] = []
    stage_identities = []
    # A curriculum may replace the base retrieval profile from optimizer step 1.
    # Step 0 is the ordering sentinel so the first explicit stage can start at 1.
    previous_start = 0
    for stage in profile.curriculum:
        unknown_stage_keys = sorted(set(stage) - {"start_step", "retrieval_config"})
        if unknown_stage_keys:
            raise ValueError("Unknown curriculum stage key(s): " + ", ".join(unknown_stage_keys))
        start_step = int(stage.get("start_step", 0))
        if start_step <= previous_start or start_step >= (profile.budget.max_optimizer_steps or 0):
            raise ValueError("Curriculum start_step values must increase and fall before the final optimizer step")
        config_values = dict(stage.get("retrieval_config", {}))
        unknown = sorted(set(config_values) - set(RetrievalConfig.__dataclass_fields__))
        if unknown:
            raise ValueError("Unknown curriculum RetrievalConfig key(s): " + ", ".join(unknown))
        for key in ("sequence_lengths", "key_counts", "query_distances", "overwrite_counts"):
            if key in config_values:
                config_values[key] = tuple(config_values[key])
        stage_config = replace(profile.retrieval, **config_values)
        stage_set = RetrievalDataset(
            task=task, examples=stage_config.train_examples, config=stage_config,
            seed=seeds.data_seed, split="train",
        )
        stage_manifest = stage_set.manifest()
        curriculum_loaders.append((start_step, retrieval_dataloader(stage_set, profile.budget.batch_size)))
        stage_identities.append({"start_step": start_step, "manifest": stage_manifest})
        previous_start = start_step
    if stage_identities:
        atomic_write_json(run_dir / "curriculum_train.json", {"stages": stage_identities})
        identity["curriculum"] = [
            {"start_step": entry["start_step"], "hash": entry["manifest"]["hash"]}
            for entry in stage_identities
        ]
    return retrieval_dataloader(train_set, profile.budget.batch_size), retrieval_dataloader(validation_retrieval, profile.budget.batch_size), identity, held_out, validation_retrieval, curriculum_loaders


def _manifest(
    *, profile: ExperimentConfig, family: Family, scale: str, variant: str, resolved_model: ModelConfig,
    model_metadata: dict[str, Any], seeds: SeedBundle, data_identity: dict[str, Any],
) -> SeedManifest:
    return SeedManifest(
        family=family, run_mode=profile.run_mode, scale=scale, variant=variant,
        model={**resolved_model.to_dict(), **model_metadata}, seeds=seeds.to_dict(),
        budget={
            "max_optimizer_steps": profile.budget.max_optimizer_steps,
            "grad_accum_steps": profile.budget.grad_accum_steps,
            "batch_size": profile.budget.batch_size,
            "context_length": profile.budget.context_length,
            "status": profile.budget.status,
            "concurrent_training_runs": profile.concurrent_training_runs,
        },
        data_identity=data_identity,
        evaluation_protocol={
            "version": 2,
            "cache_benchmarks": profile.enable_cache_benchmarks,
            "benchmark_warmup": profile.benchmark_warmup,
            "benchmark_repeats": profile.benchmark_repeats,
            "benchmark_prompt_lengths": list(profile.benchmark_prompt_lengths),
            "benchmark_generated_tokens": profile.benchmark_generated_tokens,
            "equivalence_prompt_length": profile.equivalence_prompt_length,
            "equivalence_generated_tokens": profile.equivalence_generated_tokens,
            "resource_capture_version": 1,
            "resource_sample_interval_seconds": 1.0,
            "gate_diagnostics": profile.enable_gate_diagnostics,
            "posthoc_gate_grouping": profile.enable_posthoc_gate_grouping,
            "posthoc_gate_group_counts": list(profile.posthoc_gate_group_counts),
            "posthoc_gate_eval_examples": profile.posthoc_gate_eval_examples,
            "energy": profile.enable_energy_collection,
            "retrieval_version": profile.retrieval.generator_version if family != "tinystories" else None,
            "natural_language_retrieval": "exploratory_zero_shot" if family == "tinystories" else None,
        },
        expected_artifacts=("manifest.json", "lifecycle.json", "metrics.json", "final_checkpoint.pt"),
        protocol_version=profile.protocol_version,
    )


def _load_final_checkpoint(
    model: torch.nn.Module, run_dir: Path, profile: ExperimentConfig, processed_tokens: int,
    device: torch.device,
) -> None:
    valid, reason, checkpoint = validate_checkpoint(
        run_dir / "final_checkpoint.pt", profile.budget.max_optimizer_steps or -1, processed_tokens,
    )
    if not valid or checkpoint is None:
        raise RuntimeError(f"Cannot recover evaluation from final checkpoint: {reason}")
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])


def _evenly_spaced_indices(dataset_size: int, requested_count: int) -> tuple[int, ...]:
    count = min(dataset_size, requested_count)
    if count < 1:
        raise ValueError("Post-hoc gate evaluation needs a non-empty retrieval set")
    if count == 1:
        return (dataset_size // 2,)
    return tuple(round(index * (dataset_size - 1) / (count - 1)) for index in range(count))


def _gate_group_sensitivity(
    grouped_results: dict[str, Any], family: Family, full_channel_group: int,
) -> dict[str, Any]:
    """Compare each inference intervention to the unmodified same-checkpoint result."""
    reference = grouped_results[str(full_channel_group)]
    sensitivity: dict[str, Any] = {}
    for group_name, current in grouped_results.items():
        if family == "tinystories":
            sensitivity[group_name] = {
                "validation_loss_delta_vs_full_channel": current["loss"] - reference["loss"],
                "validation_perplexity_delta_vs_full_channel": current["perplexity"] - reference["perplexity"],
            }
            continue

        overall: dict[str, float] = {}
        metric_names = ["exact_query_accuracy"]
        if family == "overwrite":
            metric_names.extend((
                "latest_value_accuracy", "stale_value_error_rate",
                "unrelated_association_retention",
            ))
        for metric in metric_names:
            value, baseline = current.get(metric), reference.get(metric)
            if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                overall[f"{metric}_delta_vs_full_channel"] = float(value) - float(baseline)

        bucket_deltas: dict[str, Any] = {}
        for dimension, current_buckets in current.get("breakdowns", {}).items():
            if not current_buckets:
                continue
            baseline_buckets = reference.get("breakdowns", {}).get(dimension, {})
            bucket_deltas[dimension] = {}
            for bucket, metrics in current_buckets.items():
                baseline_metrics = baseline_buckets.get(bucket)
                if baseline_metrics is None:
                    continue
                bucket_deltas[dimension][bucket] = {
                    "accuracy_delta_vs_full_channel": metrics["accuracy"] - baseline_metrics["accuracy"],
                    "examples_or_answers": metrics["total"],
                }
        sensitivity[group_name] = {
            "overall_metric_deltas": overall,
            "accuracy_deltas_by_difficulty": bucket_deltas,
        }
    return sensitivity


def _posthoc_gate_grouping(
    *, profile: ExperimentConfig, family: Family, model: torch.nn.Module,
    validation_data: Any, held_out_retrieval: RetrievalDataset | None, device: torch.device,
) -> dict[str, Any]:
    if not profile.enable_posthoc_gate_grouping or profile.run_mode != "production":
        return {"status": "disabled"}
    if not isinstance(model, torch.nn.Module) or not hasattr(model, "set_runtime_gate_groups"):
        raise TypeError("Post-hoc gate grouping requires a GatedDeltaNetLM")
    if getattr(model, "variant", None) != "channel-wise":
        return {"status": "not_applicable", "reason": "only channel-wise checkpoints have multiple gate channels"}

    head_dim = int(model.model_config.d_head)
    requested_groups = tuple(profile.posthoc_gate_group_counts)
    group_counts = sorted({
        groups for groups in (*requested_groups, head_dim)
        if isinstance(groups, int) and 1 <= groups <= head_dim and head_dim % groups == 0
    })
    excluded_groups = [groups for groups in requested_groups if groups not in group_counts]
    grouped_results: dict[str, Any] = {}
    indices: tuple[int, ...] | None = None
    if family != "tinystories":
        if held_out_retrieval is None:
            raise ValueError("Retrieval group evaluation requires the held-out dataset")
        indices = _evenly_spaced_indices(len(held_out_retrieval), profile.posthoc_gate_eval_examples)

    try:
        for groups in group_counts:
            model.set_runtime_gate_groups(groups)
            if family == "tinystories":
                max_steps = min(
                    profile.budget.validation_steps,
                    max(1, (profile.posthoc_gate_eval_examples + profile.budget.batch_size - 1)
                        // profile.budget.batch_size),
                )
                grouped_results[str(groups)] = evaluate_model(
                    model, validation_data, device, max_steps=max_steps,
                )
            else:
                assert held_out_retrieval is not None and indices is not None
                grouped_results[str(groups)] = evaluate_retrieval(
                    model, held_out_retrieval, device, example_indices=indices,
                )
    finally:
        model.set_runtime_gate_groups(None)

    return {
        "status": "complete",
        "evaluation_type": "posthoc_gate_sharing_intervention_no_retraining",
        "head_dimension": head_dim,
        "group_counts_per_head": group_counts,
        "excluded_non_divisor_group_counts": excluded_groups,
        "requested_retrieval_examples": profile.posthoc_gate_eval_examples,
        "evaluated_retrieval_examples": len(indices) if indices is not None else None,
        "metrics_by_group_count": grouped_results,
        "sensitivity_vs_full_channel": _gate_group_sensitivity(grouped_results, family, head_dim),
        "sensitivity_interpretation": (
            "For retrieval, negative accuracy deltas mean lower accuracy than the full-channel setting; "
            "positive stale-value-error deltas mean worse overwrite recall. These are same-checkpoint, "
            "same-example point differences, not results from separately trained group-gated models."
        ),
        "aggregation_conventions": {
            "mean_alpha": "arithmetic mean of effective retention factors alpha within each group",
            "mean_log_decay": "geometric mean of effective retention factors, implemented as exp(mean(log(alpha)))",
            "training": "no retraining; same fully channel-wise checkpoint and identical evaluation cases",
            "runtime_claim": "inference sensitivity only; projection cost and throughput are not measured by this intervention",
        },
        "interpretation_note": (
            "This measures sensitivity of a fully channel-wise trained checkpoint to sharing its "
            "gate values at inference; it does not estimate how a model trained with fewer groups performs."
        ),
    }


def _evaluate(
    *, profile: ExperimentConfig, family: Family, model: torch.nn.Module, validation_data: Any,
    held_out_retrieval: RetrievalDataset | None, device: torch.device, energy_provider: EnergyProvider,
    run_dir: Path, evaluation_seed: int,
) -> dict[str, Any]:
    assert isinstance(model, torch.nn.Module)
    gdn_model = model  # dedicated name kept to make timing/diagnostic separation obvious
    vocab_size = profile.vocab_size if family == "tinystories" else profile.retrieval.vocabulary_size
    if profile.enable_cache_benchmarks and profile.concurrent_training_runs == 1:
        benchmarks = benchmark_inference(
            gdn_model, profile, device, energy_provider, vocab_size=vocab_size,
            seed=evaluation_seed, measurement_concurrency=1, isolated=True,
        )
    elif profile.enable_cache_benchmarks:
        benchmarks = {
            "disabled": True,
            "status": "unavailable",
            "reason": "publishable inference benchmarks require isolated GPU access; evaluation workers were concurrent",
            "measurement_contract": {
                "required_concurrency": 1,
                "observed_concurrency": profile.concurrent_training_runs,
                "isolated_gpu": False,
            },
        }
    else:
        generator = torch.Generator(device=device).manual_seed(evaluation_seed)
        prompt_length = min(profile.equivalence_prompt_length, profile.budget.context_length)
        prompt = torch.randint(0, vocab_size, (1, prompt_length), generator=generator, device=device)
        benchmarks = {
            "disabled": True,
            "cache_equivalence": generation_cache_equivalence(
                gdn_model, prompt, min(profile.equivalence_generated_tokens, profile.budget.context_length),
            ),
        }
    if family == "tinystories":
        diagnostic_batches = [next(iter(validation_data))[0]]
        gate = collect_gate_diagnostics(gdn_model, diagnostic_batches, device) if profile.enable_gate_diagnostics else {"disabled": True}
        needle = {"label": "exploratory_zero_shot", "not_run_in_smoke": profile.run_mode == "smoke"}
        if profile.run_mode == "production":
            suite = persisted_needle_suite(run_dir.parent.parent.parent / "needle_suite_v1.json", seed=evaluation_seed)
            needle = evaluate_needle_suite(gdn_model, suite, device)
        static_cost = static_cost_report(gdn_model, batch_size=profile.budget.batch_size, context_length=profile.budget.context_length)
        static_cost["recurrence"] = backend_metadata(recurrence_options_for(profile), device=device)
        result = {
            "benchmarks": benchmarks, "gate_diagnostics": gate,
            "static_cost": static_cost,
            "natural_language_retrieval": needle,
        }
        result["posthoc_gate_grouping"] = _posthoc_gate_grouping(
            profile=profile, family=family, model=gdn_model, validation_data=validation_data,
            held_out_retrieval=held_out_retrieval, device=device,
        )
        return result
    assert held_out_retrieval is not None
    diagnostic_batches = [held_out_retrieval[0][0].unsqueeze(0)]
    static_cost = static_cost_report(gdn_model, batch_size=profile.budget.batch_size, context_length=profile.budget.context_length)
    static_cost["recurrence"] = backend_metadata(recurrence_options_for(profile), device=device)
    result = {
        "benchmarks": benchmarks,
        "retrieval": evaluate_retrieval(gdn_model, held_out_retrieval, device),
        "gate_diagnostics": collect_gate_diagnostics(gdn_model, diagnostic_batches, device) if profile.enable_gate_diagnostics else {"disabled": True},
        "static_cost": static_cost,
    }
    result["posthoc_gate_grouping"] = _posthoc_gate_grouping(
        profile=profile, family=family, model=gdn_model, validation_data=validation_data,
        held_out_retrieval=held_out_retrieval, device=device,
    )
    return result


def run_seed(
    profile: ExperimentConfig, *, family: Family, scale: str, variant: str, seed: int,
    stop_after_trained: bool = False, evaluate_only: bool = False,
) -> dict[str, Any]:
    """Run/recover one isolated seed, refusing stale or partial artifacts as complete."""

    if profile.run_mode == "production":
        options = recurrence_options_for(profile)
        head_dims = tuple(sorted({profile.model_for(scale).d_head for scale in profile.scales}))
        preflight = require_production_cuda(
            recurrence_backend=options.backend,
            input_dtype=options.input_dtype,
            state_accumulator_dtype=options.state_accumulator_dtype,
            chunk_size=options.chunk_size,
            qualification_artifact=options.qualification_artifact,
            head_dims=head_dims,
        )
        ensure_backend_available(options, device=torch.device("cuda"))
        profile.budget.require_ready()
    run_dir = _run_path(profile, family, scale, variant, seed)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            stored_protocol = read_json(manifest_path).get("protocol_version", LEGACY_PROTOCOL_VERSION)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Existing artifact has an unreadable manifest; refusing to modify {run_dir}") from exc
        if stored_protocol != profile.protocol_version:
            raise RuntimeError(
                "Existing artifact uses protocol "
                f"{stored_protocol!r}; requested {profile.protocol_version!r}. "
                "Start the corrected experiment in a fresh output root."
            )
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to write into non-empty run directory without a manifest: {run_dir}. "
            "Use a fresh output root or recover the incomplete artifact explicitly."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = SeedBundle.from_research_seed(seed)
    train_data, validation_data, data_identity, held_out_retrieval, validation_retrieval, curriculum_loaders = _data_for_run(profile, family, seeds, run_dir)
    seed_everything(seeds.model_seed)
    backbone = _resolved_backbone(profile, family, scale)
    vocab_size = profile.vocab_size if family == "tinystories" else profile.retrieval.vocabulary_size
    construct_kwargs = {
        "recurrence_options": recurrence_options_for(profile),
        "gate_initialization": profile.initialization,
        "study_init_seed": seeds.model_seed if profile.experiment_id is not None else None,
    }
    if variant == "grouped-4":
        construct_kwargs["gate_groups"] = 4
    model, resolved_model, model_metadata = construct_model(
        backbone, variant, vocab_size, **construct_kwargs,
    )
    calibration_path = run_dir / "calibration.json"
    if profile.initialization == "calibrated-geometric":
        if evaluate_only or manifest_path.exists():
            if not calibration_path.is_file():
                raise RuntimeError("Calibrated evaluation requires its saved calibration provenance")
            calibration_report = read_json(calibration_path)
        else:
            from gate_initialization import calibrate_gate_initialization
            if profile.calibration_min_half_life is None or profile.calibration_max_half_life is None:
                raise ValueError("Calibrated initialization requires minimum and maximum half-life")
            calibration_device = torch.device("cuda" if profile.run_mode == "production" else "cpu")
            model.to(calibration_device)
            example_limit = max(1, profile.calibration_examples)
            batch_count = (example_limit + profile.budget.batch_size - 1) // profile.budget.batch_size
            batches = list(islice(iter(train_data), batch_count))
            remaining = example_limit
            bounded_batches = []
            for batch in batches:
                tokens, targets = batch
                take = min(remaining, tokens.shape[0])
                bounded_batches.append((tokens[:take], targets[:take]))
                remaining -= take
                if remaining <= 0:
                    break
            calibration_started = time.perf_counter()
            calibration_report = calibrate_gate_initialization(
                model, bounded_batches, calibration_device,
                min_half_life=profile.calibration_min_half_life,
                max_half_life=profile.calibration_max_half_life,
            )
            calibration_report["calibration_examples"] = example_limit - remaining
            calibration_report["calibration_wall_time_seconds"] = time.perf_counter() - calibration_started
            atomic_write_json(calibration_path, calibration_report)
        model_metadata["calibration"] = calibration_report
    elif calibration_path.exists() and not evaluate_only:
        raise RuntimeError("Default initialization run directory contains stale calibration provenance")
    if profile.experiment_id is not None:
        model_metadata["study_task"] = {
            "experiment_id": profile.experiment_id,
            "condition_id": profile.condition_id,
            "family": family,
            "scale": scale,
            "variant": variant,
            "initialization": profile.initialization,
            "seed": seed,
            "budget": {
                "max_optimizer_steps": profile.budget.max_optimizer_steps,
                "grad_accum_steps": profile.budget.grad_accum_steps,
                "batch_size": profile.budget.batch_size,
                "context_length": profile.budget.context_length,
                "validation_steps": profile.budget.validation_steps,
                "evaluation_interval": profile.budget.evaluation_interval,
                **({"validation_at_steps": list(profile.budget.validation_at_steps)} if profile.budget.validation_at_steps is not None else {}),
            },
            "learning_rate": profile.learning_rate,
            "calibration_examples": profile.calibration_examples,
            "protocol_version": profile.protocol_version,
        }
    manifest = _manifest(
        profile=profile, family=family, scale=scale, variant=variant, resolved_model=resolved_model,
        model_metadata=model_metadata, seeds=seeds, data_identity=data_identity,
    )
    if manifest_path.exists():
        compatible, reason = manifest_compatibility(
            manifest_path, manifest,
            allow_source_drift=evaluate_only,
        )
        if not compatible:
            raise RuntimeError(f"Existing artifact is incompatible and will not be overwritten: {reason}")
    else:
        if evaluate_only:
            raise RuntimeError(f"Cannot evaluate missing study manifest at {manifest_path}")
        manifest_data = write_manifest(manifest_path, manifest)
        initialize_lifecycle(run_dir, manifest_data["fingerprint"])
    # Keep the fingerprint already attached to this artifact set. Evaluation may
    # accept source-only drift, but lifecycle transitions must continue using
    # the fingerprint stored with the original manifest/checkpoint.
    manifest_data = read_json(manifest_path)
    allow_source_drift = evaluate_only
    complete, _ = validate_completion(run_dir, manifest, allow_source_drift=allow_source_drift)
    if complete:
        return {"status": "skipped_complete", "run_dir": str(run_dir)}

    lifecycle = read_json(run_dir / "lifecycle.json")
    status = lifecycle["status"]
    if evaluate_only and status not in {"trained", "evaluating", "complete", "failed"}:
        raise RuntimeError(
            f"Cannot evaluate {family}/{scale}/{variant}/seed {seed}: run status is {status!r}; "
            "run the matching --train-only command first."
        )
    if status == "trained":
        try:
            partial = read_json(run_dir / "metrics.json")
            processed_tokens = int(partial["training"]["processed_tokens"])
            checkpoint_info = partial["final_checkpoint"]
            valid, reason, _ = validate_checkpoint(
                run_dir / str(checkpoint_info["path"]), profile.budget.max_optimizer_steps or -1,
                processed_tokens,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Trained study artifact is incomplete: {exc}") from exc
        if not valid:
            raise RuntimeError(f"Trained study checkpoint is invalid: {reason}")
        if stop_after_trained:
            return {"status": "trained", "run_dir": str(run_dir), "checkpoint": checkpoint_info}
    if status == "training":
        raise RuntimeError(
            "Training was interrupted without a resumable optimizer/scheduler/scaler/RNG/data-position checkpoint; "
            "start a fresh task output directory."
        )
    device = torch.device("cuda" if profile.run_mode == "production" else "cpu")
    energy_provider = EnergyProvider.create(device) if profile.enable_energy_collection else EnergyProvider(False, "energy collection disabled", {"device_type": device.type})
    try:
        if evaluate_only and status == "failed":
            # Failed evaluations retain partial training metrics. Permit recovery
            # only when those metrics identify a valid final-budget checkpoint.
            try:
                partial = read_json(run_dir / "metrics.json")
                processed_tokens = int(partial["training"]["processed_tokens"])
                checkpoint_info = partial["final_checkpoint"]
                valid, reason, _ = validate_checkpoint(
                    run_dir / str(checkpoint_info["path"]), profile.budget.max_optimizer_steps or -1,
                    processed_tokens,
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Cannot retry evaluation: completed-training artifacts are invalid: {exc}") from exc
            if not valid or checkpoint_info.get("completed_steps") != (profile.budget.max_optimizer_steps or -1):
                raise RuntimeError(f"Cannot retry evaluation: final training checkpoint is invalid: {reason}")
            transition_lifecycle(run_dir, "evaluating", manifest_data["fingerprint"], details={"retry_after_evaluation_failure": True})
            status = "evaluating"
        if not evaluate_only and status in {"pending", "training"}:
            if status == "pending":
                transition_lifecycle(run_dir, "training", manifest_data["fingerprint"])
            with DeviceResourceCapture(device) as resource_capture:
                validation_metrics_fn = None
                if profile.study_validation_answer_accuracy and validation_retrieval is not None:
                    def validation_metrics_fn(current_model: torch.nn.Module, step: int) -> dict[str, Any]:
                        indices = _evenly_spaced_indices(
                            len(validation_retrieval), min(
                                profile.retrieval.validation_examples or profile.retrieval.held_out_examples,
                                len(validation_retrieval),
                            ),
                        )
                        answer_metrics = evaluate_retrieval(
                            current_model, validation_retrieval, device, example_indices=indices,
                        )
                        return {
                            "answer_accuracy": answer_metrics["exact_query_accuracy"],
                            "answer_examples": answer_metrics["evaluated_examples"],
                            "answer_validation_seed": seeds.evaluation_seed,
                            "qualification_scope": profile.study_qualification_scope,
                        }
                training = train_fixed_budget(
                    model=model, train_dataloader=train_data, validation_dataloader=validation_data,
                    profile=profile, device=device, run_dir=str(run_dir), energy_provider=energy_provider,
                    progress_label=f"{family}/{scale}/{variant}/seed{seed}",
                    validation_metrics_fn=validation_metrics_fn,
                    curriculum_loaders=curriculum_loaders,
                )
            training_resources = resource_capture.result
            partial_metrics = {
                "schema_version": 1, "family": family, "scale": scale, "variant": variant,
                "seed": seed, "parameter": model_metadata,
                "execution": {
                    "concurrent_training_runs_limit": profile.concurrent_training_runs,
                    "telemetry_scope": (
                        "per-process CPU plus whole-device GPU/system aggregates; other training workers may overlap"
                        if profile.concurrent_training_runs > 1 else
                        "single training run on the device"
                    ),
                    "training_wall_time_note": (
                        "The launcher may overlap this run with sibling GPU work; compare wall time only with the same concurrency policy."
                        if profile.concurrent_training_runs > 1 else
                        "Measured without concurrent training workers in this launcher."
                    ),
                    "energy_measurement_note": (
                        "NVML energy/power readings cover the whole GPU while sibling runs may be active; do not sum per-run values."
                        if profile.concurrent_training_runs > 1 else
                        "Energy/power readings cover the device during this single run."
                    ),
                },
                "recurrence": {
                    **backend_metadata(recurrence_options_for(profile), device=device),
                    "qualification_hash": qualification_hash(recurrence_options_for(profile).qualification_artifact),
                    "cuda_preflight": preflight.to_dict() if profile.run_mode == "production" else None,
                },
                **training,
                "training_resources": training_resources,
            }
            atomic_write_json(run_dir / "metrics.json", partial_metrics)
            transition_lifecycle(run_dir, "trained", manifest_data["fingerprint"])
            status = "trained"
            if stop_after_trained:
                return {"status": "trained", "run_dir": str(run_dir)}
        if status in {"trained", "evaluating"}:
            partial_metrics = read_json(run_dir / "metrics.json")
            processed_tokens = int(partial_metrics["training"]["processed_tokens"])
            _load_final_checkpoint(model, run_dir, profile, processed_tokens, device)
            if status == "trained":
                transition_lifecycle(run_dir, "evaluating", manifest_data["fingerprint"])
            with DeviceResourceCapture(device) as evaluation_capture:
                evaluation = _evaluate(
                    profile=profile, family=family, model=model, validation_data=validation_data,
                    held_out_retrieval=held_out_retrieval, device=device, energy_provider=energy_provider,
                    run_dir=run_dir, evaluation_seed=seeds.evaluation_seed,
                )
            evaluation["evaluation_resources"] = evaluation_capture.result
            energy = {"training": partial_metrics["training_energy"], "device": energy_provider.device_metadata}
            metrics = {**partial_metrics, **evaluation, "energy": energy}
            atomic_write_json(run_dir / "metrics.json", metrics)
            transition_lifecycle(run_dir, "complete", manifest_data["fingerprint"])
            return {"status": "complete", "run_dir": str(run_dir)}
        raise RuntimeError(f"Cannot resume lifecycle status {status!r}")
    except Exception as exc:
        if read_json(run_dir / "lifecycle.json").get("status") != "complete":
            transition_lifecycle(run_dir, "failed", manifest_data["fingerprint"], details={"type": type(exc).__name__, "message": str(exc)})
        raise


def run_family(
    profile: ExperimentConfig, family: Family, *, stop_after_trained: bool = False,
    evaluate_only: bool = False,
) -> list[dict[str, Any]]:
    if family not in profile.enabled_families:
        return []
    results = []
    for scale in profile.scales:
        for variant in profile.variants:
            for seed in profile.seeds:
                results.append(run_seed(
                    profile, family=family, scale=scale, variant=variant, seed=seed,
                    stop_after_trained=stop_after_trained, evaluate_only=evaluate_only,
                ))
    return results


def run_selected(
    profile: ExperimentConfig, families: tuple[Family, ...] | None = None, *,
    stop_after_trained: bool = False, evaluate_only: bool = False, write_report: bool = True,
) -> dict[str, Any]:
    families = families or profile.enabled_families
    results = {
        family: run_family(profile, family, stop_after_trained=stop_after_trained, evaluate_only=evaluate_only)
        for family in families
    }
    if write_report:
        results["report"] = write_first_submission_reports(profile.output_dir, expected_run_mode=profile.run_mode)
    return results


def run_cpu_smoke(profile: ExperimentConfig) -> dict[str, Any]:
    """Exercise each required path, including evaluation-only recovery, offline."""

    if profile.run_mode != "smoke":
        raise ValueError("run_cpu_smoke requires the smoke profile")
    started = time.perf_counter()
    # Recovery is run first for one TinyStories-compatible toy seed. The second
    # invocation must load the final checkpoint and evaluate without retraining.
    recovery = run_seed(profile, family="tinystories", scale=profile.scales[0], variant="scalar", seed=profile.seeds[0], stop_after_trained=True)
    resumed = run_seed(profile, family="tinystories", scale=profile.scales[0], variant="scalar", seed=profile.seeds[0])
    results = run_selected(profile)
    elapsed = time.perf_counter() - started
    try:
        import psutil
        measured_rss_mb: float | None = psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        measured_rss_mb = None
    smoke = {
        "status": "complete", "recovery_first_pass": recovery, "recovery_resume": resumed,
        "elapsed_seconds": elapsed, "resource_target": {"seconds": 300, "ram_gb": 2},
        "measured_rss_mb_at_completion": measured_rss_mb,
        "cuda_required": False, "network_required": False,
    }
    atomic_write_json(profile.output_dir / "smoke_result.json", smoke)
    if elapsed > 300:
        raise RuntimeError(f"CPU smoke exceeded five-minute target ({elapsed:.1f}s)")
    if measured_rss_mb is not None and measured_rss_mb > 2 * 1024:
        raise RuntimeError(f"CPU smoke exceeded 2-GB RSS target ({measured_rss_mb:.1f} MB)")
    return {"smoke": smoke, **results}
