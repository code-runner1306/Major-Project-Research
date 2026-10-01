"""Configuration profiles for the typed decision fine-tuning study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Variant = Literal["scalar", "channel-wise"]
WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str = "LocalLLaMA/typed-decisions"
    dataset_revision: str = "main"
    workflows: tuple[str, ...] = WORKFLOWS
    split_seed: int = 2026
    train_cases_per_workflow: int = 240
    validation_cases_per_workflow: int = 30
    calibration_cases_per_workflow: int = 30
    expected_questions_per_case: int = 5
    tokenizer_name: str = "gpt2"
    max_length: int = 512
    truncation_policy: str = "state_head_tail"
    minimum_untruncated_fraction: float = 0.99

    def __post_init__(self) -> None:
        if not self.workflows or len(set(self.workflows)) != len(self.workflows):
            raise ValueError("workflows must be a non-empty sequence of unique names")
        if min(self.train_cases_per_workflow, self.validation_cases_per_workflow,
               self.calibration_cases_per_workflow, self.expected_questions_per_case,
               self.max_length) < 1:
            raise ValueError("case counts, question count, and max_length must be positive")
        if not 0 < self.minimum_untruncated_fraction <= 1:
            raise ValueError("minimum_untruncated_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ModelConfig:
    scale: str = "30M"
    projection_dim: int = 128
    vocab_size: int = 50_257
    scalar_checkpoint: Path = Path(
        "output/tinystories/30M/scalar/seed_43_ef7dde5a783e/final_checkpoint.pt"
    )
    channel_checkpoint: Path = Path(
        "output/tinystories/30M/channel-wise/seed_42_ced68e206c8b/final_checkpoint.pt"
    )

    def checkpoint_for(self, variant: Variant) -> Path:
        return self.scalar_checkpoint if variant == "scalar" else self.channel_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    fine_tuning_seed: int = 2026
    max_epochs: int = 3
    candidate_backbone_learning_rates: tuple[float, ...] = (1e-5, 2e-5)
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    physical_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    warmup_fraction: float = 0.05
    checkpoint_every_steps: int = 100
    validation_every_steps: int = 100
    precision: Literal["fp16", "fp32"] = "fp16"

    @property
    def effective_batch_size(self) -> int:
        return self.physical_batch_size * self.gradient_accumulation_steps

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.physical_batch_size < 1:
            raise ValueError("epochs and physical batch size must be positive")
        if self.gradient_accumulation_steps < 1 or self.checkpoint_every_steps < 1:
            raise ValueError("accumulation and checkpoint interval must be positive")
        if self.validation_every_steps < 1 or self.gradient_clip_norm <= 0:
            raise ValueError("validation interval and gradient clip must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not self.candidate_backbone_learning_rates or any(
            rate <= 0 for rate in self.candidate_backbone_learning_rates
        ):
            raise ValueError("candidate learning rates must be positive")


@dataclass(frozen=True)
class EvaluationConfig:
    calibration_seed: int = 2027
    ece_bins: int = 15
    option_permutations: int = 3
    bootstrap_replicates: int = 10_000
    bootstrap_seed: int = 2028
    latency_warmups: int = 20
    latency_repeats: int = 200

    def __post_init__(self) -> None:
        if min(self.ece_bins, self.option_permutations, self.bootstrap_replicates,
               self.latency_warmups, self.latency_repeats) < 1:
            raise ValueError("evaluation counts must be positive")


@dataclass(frozen=True)
class BaselineConfig:
    model_id: str = "convaiinnovations/laya-typed-decisions"
    revision: str = "main"
    package_name: str = "laya"
    package_version: str = "0.3.21"
    require_direct_execution: bool = True


@dataclass(frozen=True)
class ArtifactConfig:
    output_dir: Path = Path("output/decision_research")
    local_cache_dir: Path = Path("/content/decision_research_cache")
    run_fingerprint_schema: int = 1
    resume_slots: tuple[str, str] = ("A", "B")
    result_archive_name: str = "decision_research_results.tar.gz"


@dataclass(frozen=True)
class ResearchConfig:
    output_dir: Path = Path("output/decision_research")
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    baseline: BaselineConfig = BaselineConfig()
    artifacts: ArtifactConfig = ArtifactConfig()
    mode: Literal["smoke", "production"] = "production"
    config_schema_version: int = 1

    def canonical_dict(self) -> dict[str, object]:
        """Return a JSON-ready configuration with stable primitive values."""
        value = asdict(self)
        value["output_dir"] = str(self.output_dir)
        value["model"]["scalar_checkpoint"] = str(self.model.scalar_checkpoint)  # type: ignore[index]
        value["model"]["channel_checkpoint"] = str(self.model.channel_checkpoint)  # type: ignore[index]
        return value


DEFAULT_RESEARCH_CONFIG = ResearchConfig()
