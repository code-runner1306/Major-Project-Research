"""Shared strict orchestration for TinyStories, MQAR, and overwrite recall."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
from typing import Any, Literal

import torch

from artifacts import (
    SeedManifest, atomic_write_json, fingerprint, initialize_lifecycle, manifest_compatibility,
    read_json, transition_lifecycle, validate_completion, validate_checkpoint, write_manifest,
)
from config import ExperimentConfig, ModelConfig
from controls import construct_model
from dataset import tinystories_dataloader, toy_dataloader
from diagnostics import collect_gate_diagnostics, static_cost_report
from energy import EnergyProvider
from eval import benchmark_inference, evaluate_needle_suite, generation_cache_equivalence, persisted_needle_suite
from preflight import require_production_cuda
from recurrence import RecurrenceOptions, backend_metadata, ensure_backend_available, qualification_hash
from reporting import write_first_submission_reports
from retrieval import RetrievalDataset, evaluate_retrieval, retrieval_dataloader
from resource_telemetry import DeviceResourceCapture
from train import SeedBundle, seed_everything, train_fixed_budget


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
    identity = {
        "run_mode": profile.run_mode, "family": family, "scale": scale, "variant": variant,
        "profile": profile.to_manifest_dict(), "seed": seed,
    }
    return profile.output_dir / family / scale / variant / f"seed_{seed}_{fingerprint(identity)[:12]}"


def _resolved_backbone(profile: ExperimentConfig, family: Family, scale: str) -> ModelConfig:
    backbone = profile.model_for(scale)
    return replace(backbone, attention_layout="pure_gdn") if family in {"mqar", "overwrite"} else backbone


def _data_for_run(
    profile: ExperimentConfig, family: Family, seeds: SeedBundle, run_dir: Path
) -> tuple[Any, Any, dict[str, Any], Any | None]:
    if family == "tinystories":
        if profile.run_mode == "smoke":
            data_identity = {
                "dataset": "toy-language-v1", "data_seed": seeds.data_seed,
                "train_examples": 12, "validation_examples": 4, "paired_order": "index_order",
            }
            return toy_dataloader(profile, seed=seeds.data_seed, split="train"), toy_dataloader(profile, seed=seeds.data_seed, split="validation"), data_identity, None
        train = tinystories_dataloader(profile, split="train", data_seed=seeds.data_seed, run_dir=run_dir)
        validation = tinystories_dataloader(profile, split="validation", data_seed=seeds.data_seed, run_dir=run_dir)
        data_identity = read_json(run_dir / "data_identity.json")
        data_identity["paired_order"] = "datasets.Dataset.shuffle(seed=data_seed)"
        return train, validation, data_identity, None

    task = "mqar" if family == "mqar" else "overwrite"
    train_set = RetrievalDataset(
        task=task, examples=profile.retrieval.train_examples, config=profile.retrieval,
        seed=seeds.data_seed, split="train",
    )
    held_out = RetrievalDataset(
        task=task, examples=profile.retrieval.held_out_examples, config=profile.retrieval,
        seed=seeds.evaluation_seed, split="held_out",
    )
    train_manifest, held_manifest = train_set.manifest(), held_out.manifest()
    if train_manifest["hash"] == held_manifest["hash"]:
        raise AssertionError("Retrieval train and held-out generator identities must not overlap")
    atomic_write_json(run_dir / "generator_train.json", train_manifest)
    atomic_write_json(run_dir / "generator_held_out.json", held_manifest)
    identity = {
        "generator_version": train_manifest["generator_version"], "train_hash": train_manifest["hash"],
        "held_out_hash": held_manifest["hash"], "paired_order": "indexed_seeded_order",
    }
    return retrieval_dataloader(train_set, profile.budget.batch_size), retrieval_dataloader(held_out, profile.budget.batch_size), identity, held_out


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
            "energy": profile.enable_energy_collection,
            "retrieval_version": "mqar-overwrite-v1" if family != "tinystories" else None,
            "natural_language_retrieval": "exploratory_zero_shot" if family == "tinystories" else None,
        },
        expected_artifacts=("manifest.json", "lifecycle.json", "metrics.json", "final_checkpoint.pt"),
    )


def _load_final_checkpoint(model: torch.nn.Module, run_dir: Path, profile: ExperimentConfig, processed_tokens: int) -> None:
    valid, reason, checkpoint = validate_checkpoint(
        run_dir / "final_checkpoint.pt", profile.budget.max_optimizer_steps or -1, processed_tokens,
    )
    if not valid or checkpoint is None:
        raise RuntimeError(f"Cannot recover evaluation from final checkpoint: {reason}")
    model.load_state_dict(checkpoint["model_state"])


def _evaluate(
    *, profile: ExperimentConfig, family: Family, model: torch.nn.Module, validation_data: Any,
    held_out_retrieval: RetrievalDataset | None, device: torch.device, energy_provider: EnergyProvider,
    run_dir: Path, evaluation_seed: int,
) -> dict[str, Any]:
    assert isinstance(model, torch.nn.Module)
    gdn_model = model  # dedicated name kept to make timing/diagnostic separation obvious
    vocab_size = profile.vocab_size if family == "tinystories" else profile.retrieval.vocabulary_size
    if profile.enable_cache_benchmarks:
        benchmarks = benchmark_inference(
            gdn_model, profile, device, energy_provider, vocab_size=vocab_size, seed=evaluation_seed,
        )
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
        return {
            "benchmarks": benchmarks, "gate_diagnostics": gate,
            "static_cost": static_cost,
            "natural_language_retrieval": needle,
        }
    assert held_out_retrieval is not None
    diagnostic_batches = [held_out_retrieval[0][0].unsqueeze(0)]
    static_cost = static_cost_report(gdn_model, batch_size=profile.budget.batch_size, context_length=profile.budget.context_length)
    static_cost["recurrence"] = backend_metadata(recurrence_options_for(profile), device=device)
    return {
        "benchmarks": benchmarks,
        "retrieval": evaluate_retrieval(gdn_model, held_out_retrieval, device),
        "gate_diagnostics": collect_gate_diagnostics(gdn_model, diagnostic_batches, device) if profile.enable_gate_diagnostics else {"disabled": True},
        "static_cost": static_cost,
    }


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
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = SeedBundle.from_research_seed(seed)
    train_data, validation_data, data_identity, held_out_retrieval = _data_for_run(profile, family, seeds, run_dir)
    seed_everything(seeds.model_seed)
    backbone = _resolved_backbone(profile, family, scale)
    vocab_size = profile.vocab_size if family == "tinystories" else profile.retrieval.vocabulary_size
    model, resolved_model, model_metadata = construct_model(
        backbone, variant, vocab_size, recurrence_options=recurrence_options_for(profile),
    )
    manifest = _manifest(
        profile=profile, family=family, scale=scale, variant=variant, resolved_model=resolved_model,
        model_metadata=model_metadata, seeds=seeds, data_identity=data_identity,
    )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        compatible, reason = manifest_compatibility(manifest_path, manifest)
        if not compatible:
            raise RuntimeError(f"Existing artifact is incompatible and will not be overwritten: {reason}")
    else:
        manifest_data = write_manifest(manifest_path, manifest)
        initialize_lifecycle(run_dir, manifest_data["fingerprint"])
    manifest_data = manifest.to_dict()
    complete, _ = validate_completion(run_dir, manifest)
    if complete:
        return {"status": "skipped_complete", "run_dir": str(run_dir)}

    lifecycle = read_json(run_dir / "lifecycle.json")
    status = lifecycle["status"]
    if evaluate_only and status not in {"trained", "evaluating", "complete"}:
        raise RuntimeError(
            f"Cannot evaluate {family}/{scale}/{variant}/seed {seed}: run status is {status!r}; "
            "run the matching --train-only command first."
        )
    device = torch.device("cuda" if profile.run_mode == "production" else "cpu")
    energy_provider = EnergyProvider.create(device) if profile.enable_energy_collection else EnergyProvider(False, "energy collection disabled", {"device_type": device.type})
    try:
        if not evaluate_only and status in {"pending", "training"}:
            if status == "pending":
                transition_lifecycle(run_dir, "training", manifest_data["fingerprint"])
            with DeviceResourceCapture(device) as resource_capture:
                training = train_fixed_budget(
                    model=model, train_dataloader=train_data, validation_dataloader=validation_data,
                    profile=profile, device=device, run_dir=str(run_dir), energy_provider=energy_provider,
                )
            training_resources = resource_capture.result
            partial_metrics = {
                "schema_version": 1, "family": family, "scale": scale, "variant": variant,
                "seed": seed, "parameter": model_metadata,
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
            _load_final_checkpoint(model, run_dir, profile, processed_tokens)
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
    stop_after_trained: bool = False, evaluate_only: bool = False,
) -> dict[str, Any]:
    families = families or profile.enabled_families
    results = {
        family: run_family(profile, family, stop_after_trained=stop_after_trained, evaluate_only=evaluate_only)
        for family in families
    }
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
