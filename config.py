"""Immutable experiment profiles for the Gated DeltaNet study.

Production duration is intentionally unset until it is measured on the college
V100. The smoke profile is the only profile that is safe to run on a CPU
laptop and is selected explicitly by the command-line entry points.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
import os


Variant = Literal["scalar", "grouped-4", "channel-wise", "scalar-matched"]
ExperimentFamily = Literal["tinystories", "mqar", "overwrite"]
RecurrenceBackend = Literal["reference", "cuda-fused"]
RecurrenceInputDtype = Literal["autocast", "float32"]
StateAccumulatorDtype = Literal["input", "float32"]


@dataclass(frozen=True)
class ModelConfig:
    """Backbone settings; ``d_ff`` may be a shared width or a layer schedule."""

    name: str
    d_model: int
    n_layer: int
    n_head: int
    d_ff: int | tuple[int, ...]
    attention_layout: Literal["hybrid", "pure_gdn"] = "hybrid"

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        return self.d_model // self.n_head

    @property
    def ffn_widths(self) -> tuple[int, ...]:
        if isinstance(self.d_ff, int):
            return (self.d_ff,) * self.n_layer
        if len(self.d_ff) != self.n_layer:
            raise ValueError("per-layer d_ff schedule must have n_layer entries")
        if any(width <= 0 for width in self.d_ff):
            raise ValueError("FFN widths must be positive")
        return self.d_ff

    def to_dict(self) -> dict:
        data = asdict(self)
        data["d_ff"] = list(self.ffn_widths)
        data["d_head"] = self.d_head
        return data


@dataclass(frozen=True)
class TrainingBudget:
    """Fixed budget shared by every paired variant in one experiment family."""

    max_optimizer_steps: int | None
    grad_accum_steps: int
    batch_size: int
    context_length: int
    validation_steps: int
    evaluation_interval: int
    status: str = "pending_v100_pilot"

    def require_ready(self) -> None:
        if self.max_optimizer_steps is None:
            raise RuntimeError(
                "Production training duration is pending V100 pilot measurements. "
                "Set a fixed optimizer-step budget in a recorded production profile "
                "after the pilot; do not infer one from CPU smoke timing."
            )


@dataclass(frozen=True)
class RetrievalConfig:
    vocabulary_size: int = 8_192
    train_examples: int = 100_000
    held_out_examples: int = 3_000
    sequence_lengths: tuple[int, ...] = (128, 256, 512)
    key_counts: tuple[int, ...] = (8, 16, 32)
    query_distances: tuple[int, ...] = (16, 64, 128)
    overwrite_counts: tuple[int, ...] = (1, 2, 3)
    # Legacy experiments keep their exact v2 data identity unless a study opts in.
    generator_version: str = "mqar-overwrite-v2"
    validation_examples: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    run_mode: Literal["production", "smoke"]
    output_dir: Path
    models: dict[str, ModelConfig]
    scales: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: TrainingBudget
    vocab_size: int
    enabled_families: tuple[ExperimentFamily, ...] = ("tinystories", "mqar", "overwrite")
    enable_parameter_matched_scalar_control: bool = False
    selected_variants: tuple[Variant, ...] | None = None
    enable_cache_benchmarks: bool = True
    enable_full_prefix_benchmark: bool = False
    enable_gate_diagnostics: bool = True
    enable_posthoc_gate_grouping: bool = True
    posthoc_gate_group_counts: tuple[int, ...] = (1, 2, 4, 8)
    posthoc_gate_eval_examples: int = 512
    enable_energy_collection: bool = True
    benchmark_warmup: int = 2
    benchmark_repeats: int = 5
    benchmark_prompt_lengths: tuple[int, ...] = (64, 256, 512)
    benchmark_generated_tokens: int = 256
    equivalence_prompt_length: int = 64
    equivalence_generated_tokens: int = 256
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    recurrence_backend: RecurrenceBackend = "reference"
    recurrence_input_dtype: RecurrenceInputDtype = "autocast"
    state_accumulator_dtype: StateAccumulatorDtype = "input"
    recurrence_chunk_size: int = 64
    recurrence_qualification_artifact: Path | None = None
    concurrent_training_runs: int = 1
    # Study identity and gate initialization are independent from the historical
    # profile fields so old run identities remain stable when these are absent.
    experiment_id: str | None = None
    condition_id: str | None = None
    initialization: Literal["default", "calibrated-geometric"] = "default"
    calibration_min_half_life: float | None = None
    calibration_max_half_life: float | None = None
    calibration_examples: int = 8
    study_task_run_dir: Path | None = None
    study_qualification: bool = False
    study_qualification_stage: str | None = None
    study_qualification_scope: str | None = None
    study_validation_answer_accuracy: bool = False
    curriculum: tuple[dict, ...] = ()

    @property
    def variants(self) -> tuple[Variant, ...]:
        primary: tuple[Variant, ...] = ("scalar", "channel-wise")
        available = primary + (("scalar-matched",) if self.enable_parameter_matched_scalar_control else ())
        if self.selected_variants and "grouped-4" in self.selected_variants:
            available += ("grouped-4",)
        if self.selected_variants is None:
            return available
        return tuple(variant for variant in self.selected_variants if variant in available)

    def model_for(self, scale: str) -> ModelConfig:
        try:
            return self.models[scale]
        except KeyError as exc:
            raise ValueError(f"Unknown scale {scale!r}; expected one of {self.scales}") from exc

    def to_manifest_dict(self) -> dict:
        retrieval_data = {
            "vocabulary_size": self.retrieval.vocabulary_size,
            "train_examples": self.retrieval.train_examples,
            "held_out_examples": self.retrieval.held_out_examples,
            "sequence_lengths": list(self.retrieval.sequence_lengths),
            "key_counts": list(self.retrieval.key_counts),
            "query_distances": list(self.retrieval.query_distances),
            "overwrite_counts": list(self.retrieval.overwrite_counts),
        }
        if self.retrieval.generator_version != "mqar-overwrite-v2":
            retrieval_data["generator_version"] = self.retrieval.generator_version
        if self.retrieval.validation_examples is not None:
            retrieval_data["validation_examples"] = self.retrieval.validation_examples
        data = {
            "run_mode": self.run_mode,
            "output_dir": str(self.output_dir),
            "scales": list(self.scales),
            "models": {name: model.to_dict() for name, model in self.models.items()},
            "seeds": list(self.seeds),
            "budget": asdict(self.budget),
            "vocab_size": self.vocab_size,
            "enabled_families": list(self.enabled_families),
            "variants": list(self.variants),
            "enable_cache_benchmarks": self.enable_cache_benchmarks,
            "enable_full_prefix_benchmark": self.enable_full_prefix_benchmark,
            "enable_gate_diagnostics": self.enable_gate_diagnostics,
            "enable_posthoc_gate_grouping": self.enable_posthoc_gate_grouping,
            "posthoc_gate_group_counts": list(self.posthoc_gate_group_counts),
            "posthoc_gate_eval_examples": self.posthoc_gate_eval_examples,
            "enable_energy_collection": self.enable_energy_collection,
            "benchmark_warmup": self.benchmark_warmup,
            "benchmark_repeats": self.benchmark_repeats,
            "benchmark_prompt_lengths": list(self.benchmark_prompt_lengths),
            "benchmark_generated_tokens": self.benchmark_generated_tokens,
            "equivalence_prompt_length": self.equivalence_prompt_length,
            "equivalence_generated_tokens": self.equivalence_generated_tokens,
            "retrieval": retrieval_data,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "grad_clip": self.grad_clip,
            "recurrence": {
                "backend": self.recurrence_backend,
                "input_dtype": self.recurrence_input_dtype,
                "state_accumulator_dtype": self.state_accumulator_dtype,
                "chunk_size": self.recurrence_chunk_size,
                "qualification_artifact": (
                    str(self.recurrence_qualification_artifact)
                    if self.recurrence_qualification_artifact is not None else None
                ),
            },
            "concurrent_training_runs": self.concurrent_training_runs,
        }
        # Do not perturb historical production/smoke profile fingerprints.
        if self.experiment_id is not None:
            data["study"] = {
                "experiment_id": self.experiment_id,
                "condition_id": self.condition_id,
                "initialization": self.initialization,
                "calibration_min_half_life": self.calibration_min_half_life,
                "calibration_max_half_life": self.calibration_max_half_life,
                "calibration_examples": self.calibration_examples,
                "qualification": self.study_qualification,
                "qualification_stage": self.study_qualification_stage,
                "qualification_scope": self.study_qualification_scope,
                "validation_answer_accuracy": self.study_validation_answer_accuracy,
                "curriculum": list(self.curriculum),
            }
        return data


PRODUCTION_MODELS = {
    "10M": ModelConfig("10M", d_model=160, n_layer=6, n_head=4, d_ff=426),
    "30M": ModelConfig("30M", d_model=320, n_layer=10, n_head=8, d_ff=853),
    "55M": ModelConfig("55M", d_model=448, n_layer=12, n_head=8, d_ff=1194),
    "100M": ModelConfig("100M", d_model=576, n_layer=16, n_head=12, d_ff=1536),
}


def production_profile(output_dir: str | Path = "output") -> ExperimentConfig:
    """Return the V100-targeted profile. Its duration remains intentionally unset."""

    return ExperimentConfig(
        run_mode="production",
        output_dir=Path(output_dir),
        models=PRODUCTION_MODELS,
        scales=("10M", "30M", "55M", "100M"),
        seeds=(42, 43, 44),
        budget=TrainingBudget(
            max_optimizer_steps=None,
            grad_accum_steps=4,
            batch_size=8,
            context_length=512,
            validation_steps=200,
            evaluation_interval=100,
        ),
        vocab_size=50_257,
        enable_parameter_matched_scalar_control=os.environ.get(
            "ENABLE_PARAMETER_MATCHED_SCALAR_CONTROL", "0"
        ).lower() in {"1", "true", "yes"},
    )


def smoke_profile(output_dir: str | Path = "output/smoke") -> ExperimentConfig:
    """A CPU-only, offline profile that exercises every experiment family."""

    return ExperimentConfig(
        run_mode="smoke",
        output_dir=Path(output_dir),
        models={
            "tiny": ModelConfig(
                "tiny", d_model=32, n_layer=3, n_head=4, d_ff=64,
                attention_layout="pure_gdn",
            )
        },
        scales=("tiny",),
        seeds=(7,),
        budget=TrainingBudget(
            max_optimizer_steps=2,
            grad_accum_steps=1,
            batch_size=2,
            context_length=16,
            validation_steps=1,
            evaluation_interval=1,
            status="smoke_fixed_budget",
        ),
        vocab_size=128,
        enable_parameter_matched_scalar_control=True,
        enable_posthoc_gate_grouping=False,
        benchmark_warmup=1,
        benchmark_repeats=1,
        benchmark_prompt_lengths=(4, 8),
        benchmark_generated_tokens=2,
        equivalence_prompt_length=8,
        equivalence_generated_tokens=2,
        retrieval=RetrievalConfig(
            vocabulary_size=128,
            train_examples=8,
            held_out_examples=6,
            sequence_lengths=(16, 24),
            key_counts=(2, 3),
            query_distances=(2, 4),
            overwrite_counts=(1, 2),
        ),
    )


# Compatibility aliases for short scripts and legacy imports. New code accepts an
# ExperimentConfig explicitly and never changes these values at runtime.
DEFAULT_PROFILE = production_profile()
SCALE_CONFIGS = {
    name: {
        "d_model": model.d_model,
        "n_layer": model.n_layer,
        "n_head": model.n_head,
        "d_ff": model.d_ff,
    }
    for name, model in PRODUCTION_MODELS.items()
}
SCALES = list(DEFAULT_PROFILE.scales)
VARIANTS = list(DEFAULT_PROFILE.variants)
SEEDS = list(DEFAULT_PROFILE.seeds)
VOCAB_SIZE = DEFAULT_PROFILE.vocab_size
OUTPUT_DIR = str(DEFAULT_PROFILE.output_dir)
DATA_DIR = "data"
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
GLOBAL_TRACKER_FILE = os.path.join(OUTPUT_DIR, "global_tracker.json")
RESUME_RUN_ID = os.environ.get("RESUME_RUN_ID")
