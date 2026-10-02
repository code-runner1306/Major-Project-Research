"""Typed Decisions data preparation for the decision fine-tuning pipeline.

This module deliberately has no dependency on the model or training code.  It
owns dataset identity, case-level splits, typed-question normalization, GPT-2
serialization, padding, option permutation, and data auditing.

The official test split is guarded in two places: the loader will not request
it unless the caller explicitly asks for the final stage, and normalization
will reject a test partition unless the same explicit opt-in is supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DATASET_NAME = "LocalLLaMA/typed-decisions"
DATASET_REVISION = "main"
DATA_SCHEMA_VERSION = "typed-decision-data/v1"
SERIALIZATION_VERSION = "gpt2-state-head-tail/v1"
TOKENIZER_NAME = "gpt2"
WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)
PARTITIONS = ("train", "validation", "calibration")
PRIMITIVES = ("choice", "score", "noul")
DEFAULT_SPLIT_SEED = 2026
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_TARGET_TOLERANCE = 1e-5
DEFAULT_CASE_COUNTS = {"train": 240, "validation": 30, "calibration": 30}
DEFAULT_RELEASE_CARDINALITY = {"train": 300, "test": 100}


class DecisionDataError(ValueError):
    """Base error for incompatible or malformed decision data."""


class OfficialTestAccessError(DecisionDataError):
    """Raised when a development operation attempts to access test data."""


class ManifestMismatchError(DecisionDataError):
    """Raised when a persisted identity does not match newly prepared data."""


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return value


def canonical_json(value: Any) -> str:
    """Return the stable JSON representation used by all identity hashes."""

    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Write JSON through a same-directory temporary file and replace."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(target)


def _decode_field(value: Any, *, field_name: str, workflow: str, case_id: str) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise DecisionDataError(
            f"{workflow}/{case_id}: field {field_name!r} is not valid JSON"
        ) from exc


def _row_value(row: Mapping[str, Any], names: Sequence[str], *, field_name: str, workflow: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise DecisionDataError(f"{workflow}: row is missing {field_name}; tried {list(names)}")


def _materialize_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(rows, Mapping):
        return [dict(rows)]
    return [dict(row) for row in rows]


def _case_id(row: Mapping[str, Any], workflow: str) -> str:
    for name in ("id", "case_id", "uid"):
        if name in row and row[name] is not None:
            value = str(row[name])
            if value:
                return value
    raise DecisionDataError(f"{workflow}: row has no non-empty id/case_id/uid field")


def _state_object(row: Mapping[str, Any], workflow: str, case_id: str) -> Any:
    return _decode_field(_row_value(row, ("state",), field_name="state", workflow=workflow), field_name="state", workflow=workflow, case_id=case_id)


def canonical_text(value: Any) -> str:
    """Render a state or option value without depending on dictionary order."""

    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _state_hash(state: Any) -> str:
    return stable_hash(state)


def _row_content_hash(row: Mapping[str, Any], workflow: str, case_id: str) -> str:
    state = _state_object(row, workflow, case_id)
    questions = _decode_field(row.get("questions"), field_name="questions", workflow=workflow, case_id=case_id)
    gold = _decode_field(row.get("gold"), field_name="gold", workflow=workflow, case_id=case_id)
    return stable_hash({"workflow": workflow, "case_id": case_id, "state": state, "questions": questions, "gold": gold})


@dataclass(frozen=True)
class Option:
    label: str
    description: str
    canonical_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "canonical_index": self.canonical_index,
        }


@dataclass(frozen=True)
class DecisionExample:
    workflow: str
    case_id: str
    question_id: str
    primitive: str
    instructions: str
    state_text: str
    options: tuple[Option, ...]
    target: tuple[float, ...]
    source_partition: str
    state_hash: str
    row_hash: str
    score_values: tuple[float, ...] | None = None
    gold_label: str | None = None
    source_index: int | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(option.label for option in self.options)

    @property
    def option_count(self) -> int:
        return len(self.options)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "case_id": self.case_id,
            "question_id": self.question_id,
            "primitive": self.primitive,
            "instructions": self.instructions,
            "state_text": self.state_text,
            "options": [option.to_dict() for option in self.options],
            "target": list(self.target),
            "source_partition": self.source_partition,
            "state_hash": self.state_hash,
            "row_hash": self.row_hash,
            "score_values": list(self.score_values) if self.score_values is not None else None,
            "gold_label": self.gold_label,
            "source_index": self.source_index,
            "provenance": _jsonable(dict(self.provenance)),
        }


@dataclass(frozen=True)
class SerializedDecision:
    """Tokenized decision with half-open option spans ``[start, end)``."""

    token_ids: tuple[int, ...]
    decision_position: int
    decision_span: tuple[int, int]
    question_span: tuple[int, int]
    option_spans: tuple[tuple[int, int], ...]
    target: tuple[float, ...]
    labels: tuple[str, ...]
    options: tuple[Option, ...]
    provenance: Mapping[str, Any]
    option_token_ids: tuple[tuple[int, ...], ...] = ()
    score_values: tuple[float, ...] | None = None
    state_text: str = ""
    instructions: str = ""

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def option_count(self) -> int:
        return len(self.option_spans)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_ids": list(self.token_ids),
            "decision_position": self.decision_position,
            "decision_span": list(self.decision_span),
            "question_span": list(self.question_span),
            "option_spans": [list(span) for span in self.option_spans],
            "target": list(self.target),
            "labels": list(self.labels),
            "options": [option.to_dict() for option in self.options],
            "provenance": _jsonable(dict(self.provenance)),
            "score_values": list(self.score_values) if self.score_values is not None else None,
            "state_text": self.state_text,
            "instructions": self.instructions,
        }


@dataclass(frozen=True)
class SplitManifest:
    schema_version: str
    dataset_name: str
    dataset_revision: str
    split_seed: int
    workflows: tuple[str, ...]
    assignments: Mapping[str, Mapping[str, tuple[str, ...]]]
    state_hashes: Mapping[str, str]
    official_test_hashes: Mapping[str, str] = field(default_factory=dict)
    split_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "split_seed": self.split_seed,
            "workflows": list(self.workflows),
            "assignments": {
                workflow: {partition: list(ids) for partition, ids in partitions.items()}
                for workflow, partitions in self.assignments.items()
            },
            "state_hashes": dict(sorted(self.state_hashes.items())),
            "official_test_hashes": dict(sorted(self.official_test_hashes.items())),
        }
        payload["split_hash"] = self.split_hash or stable_hash(payload)
        return payload


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = DATASET_NAME
    dataset_revision: str = DATASET_REVISION
    split_seed: int = DEFAULT_SPLIT_SEED
    context_length: int = DEFAULT_CONTEXT_LENGTH
    tokenizer_name: str = TOKENIZER_NAME
    target_tolerance: float = DEFAULT_TARGET_TOLERANCE
    train_case_count: int = 240
    validation_case_count: int = 30
    calibration_case_count: int = 30
    expected_train_cases: int = 300
    expected_test_cases: int = 100
    expected_questions_per_case: int = 5

    @property
    def case_counts(self) -> dict[str, int]:
        return {
            "train": self.train_case_count,
            "validation": self.validation_case_count,
            "calibration": self.calibration_case_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_revision": self.dataset_revision,
            "split_seed": self.split_seed,
            "context_length": self.context_length,
            "tokenizer_name": self.tokenizer_name,
            "target_tolerance": self.target_tolerance,
            "case_counts": self.case_counts,
            "expected_train_cases": self.expected_train_cases,
            "expected_test_cases": self.expected_test_cases,
            "expected_questions_per_case": self.expected_questions_per_case,
        }


def _load_hf_dataset(name: str, workflow: str, split: str, revision: str | None) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Typed Decisions loading requires the datasets package") from exc
    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return load_dataset(name, workflow, **kwargs)


def load_workflow_rows(
    workflow: str,
    *,
    split: str = "train",
    revision: str | None = DATASET_REVISION,
    dataset_name: str = DATASET_NAME,
    stage: str = "development",
    allow_official_test: bool = False,
    loader: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Load one workflow, with an explicit guard around the official test split."""

    if workflow not in WORKFLOWS:
        raise DecisionDataError(f"Unknown Typed Decisions workflow: {workflow}")
    if split == "test" and not (allow_official_test and stage == "final"):
        raise OfficialTestAccessError(
            "The official test split requires allow_official_test=True and stage='final'"
        )
    source = loader or _load_hf_dataset
    rows = source(dataset_name, workflow, split, revision) if loader is None else source(
        dataset_name, workflow, split=split, revision=revision
    )
    return _materialize_rows(rows)


def load_all_training_workflows(
    *,
    revision: str | None = DATASET_REVISION,
    dataset_name: str = DATASET_NAME,
    loader: Callable[..., Any] | None = None,
    expected_cases: int | None = 300,
    expected_questions_per_case: int | None = 5,
) -> dict[str, list[dict[str, Any]]]:
    return {
        workflow: validate_workflow_rows(
            load_workflow_rows(
                workflow,
                split="train",
                revision=revision,
                dataset_name=dataset_name,
                loader=loader,
            ),
            workflow,
            split="train",
            expected_cases=expected_cases,
            expected_questions_per_case=expected_questions_per_case,
        )
        for workflow in WORKFLOWS
    }


def load_official_test_workflows(
    *,
    revision: str | None = DATASET_REVISION,
    dataset_name: str = DATASET_NAME,
    loader: Callable[..., Any] | None = None,
    expected_cases: int | None = 100,
    expected_questions_per_case: int | None = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Load test rows only for an explicit final-stage invocation."""

    return {
        workflow: validate_workflow_rows(
            load_workflow_rows(
                workflow,
                split="test",
                revision=revision,
                dataset_name=dataset_name,
                stage="final",
                allow_official_test=True,
                loader=loader,
            ),
            workflow,
            split="test",
            expected_cases=expected_cases,
            expected_questions_per_case=expected_questions_per_case,
        )
        for workflow in WORKFLOWS
    }


def _questions_object(row: Mapping[str, Any], workflow: str, case_id: str) -> Mapping[str, Any]:
    questions = _decode_field(row.get("questions"), field_name="questions", workflow=workflow, case_id=case_id)
    if not isinstance(questions, Mapping):
        raise DecisionDataError(f"{workflow}/{case_id}: questions must be a JSON object")
    return questions


def _gold_object(row: Mapping[str, Any], workflow: str, case_id: str) -> Mapping[str, Any]:
    gold = _decode_field(row.get("gold"), field_name="gold", workflow=workflow, case_id=case_id)
    if not isinstance(gold, Mapping):
        raise DecisionDataError(f"{workflow}/{case_id}: gold must be a JSON object")
    return gold


def validate_workflow_rows(
    rows: Iterable[Mapping[str, Any]],
    workflow: str,
    *,
    split: str = "train",
    expected_cases: int | None = 300,
    expected_questions_per_case: int | None = 5,
) -> list[dict[str, Any]]:
    materialized = _materialize_rows(rows)
    if expected_cases is not None and len(materialized) != expected_cases:
        raise DecisionDataError(
            f"{workflow}/{split}: expected {expected_cases} cases, found {len(materialized)}"
        )
    ids: set[str] = set()
    for row in materialized:
        case_id = _case_id(row, workflow)
        if case_id in ids:
            raise DecisionDataError(f"{workflow}/{split}: duplicate case id {case_id}")
        ids.add(case_id)
        questions = _questions_object(row, workflow, case_id)
        gold = _gold_object(row, workflow, case_id)
        if expected_questions_per_case is not None and len(questions) != expected_questions_per_case:
            raise DecisionDataError(
                f"{workflow}/{split}/{case_id}: expected {expected_questions_per_case} questions, found {len(questions)}"
            )
        if set(questions) != set(gold):
            raise DecisionDataError(f"{workflow}/{split}/{case_id}: question and gold ids differ")
        for question_id, question in questions.items():
            if not isinstance(question, Mapping):
                raise DecisionDataError(f"{workflow}/{split}/{case_id}/{question_id}: question is not an object")
            primitive = question.get("type", question.get("primitive"))
            if primitive not in PRIMITIVES:
                raise DecisionDataError(f"{workflow}/{split}/{case_id}/{question_id}: invalid type {primitive!r}")
    return materialized


def _ordered_labels(primitive: str, criteria: Mapping[str, Any]) -> list[str]:
    labels = [str(label) for label in criteria]
    if primitive == "noul":
        available = {label.lower(): label for label in labels}
        if {"false", "true"}.issubset(available):
            return [available["false"], available["true"]]
        return ["false", "true"]
    if primitive == "score":
        try:
            return [label for _, label in sorted((float(label), label) for label in labels)]
        except (TypeError, ValueError):
            return labels
    return labels


def _option_descriptions(question: Mapping[str, Any], labels: Sequence[str]) -> dict[str, str]:
    criteria = question.get("criteria")
    if criteria is None:
        criteria = {}
    if not isinstance(criteria, Mapping):
        raise DecisionDataError("question.criteria must be an object")
    descriptions = {str(label): canonical_text(description) for label, description in criteria.items()}
    if not descriptions and set(labels) == {"false", "true"}:
        descriptions = {"false": "The statement is false.", "true": "The statement is true."}
    for label in labels:
        descriptions.setdefault(label, "No description provided.")
    return descriptions


def _probability_mapping(
    primitive: str,
    answer: Any,
    labels: Sequence[str],
    *,
    workflow: str,
    case_id: str,
    question_id: str,
) -> Mapping[str, Any]:
    if not isinstance(answer, Mapping):
        raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: gold answer is not an object")
    probabilities = answer.get("probabilities")
    if probabilities is None and primitive == "noul":
        probability_true = answer.get("probability_true", answer.get("noul"))
        if probability_true is not None:
            probabilities = {"false": 1.0 - float(probability_true), "true": float(probability_true)}
    if probabilities is None:
        raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: missing gold probabilities")
    if not isinstance(probabilities, Mapping):
        raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: probabilities must be an object")
    available = {str(label): value for label, value in probabilities.items()}
    missing = [label for label in labels if label not in available]
    extra = [label for label in available if label not in labels]
    if missing or extra:
        raise DecisionDataError(
            f"{workflow}/{case_id}/{question_id}: probability labels mismatch; missing={missing}, extra={extra}"
        )
    return available


def validate_soft_target(
    probabilities: Mapping[str, Any],
    labels: Sequence[str],
    *,
    workflow: str = "unknown",
    case_id: str = "unknown",
    question_id: str = "unknown",
    tolerance: float = DEFAULT_TARGET_TOLERANCE,
) -> tuple[float, ...]:
    """Validate and normalize a finite non-negative target in label order."""

    values: list[float] = []
    for label in labels:
        if label not in probabilities:
            raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: missing target label {label!r}")
        try:
            value = float(probabilities[label])
        except (TypeError, ValueError) as exc:
            raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: non-numeric target for {label!r}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: invalid target value {value!r} for {label!r}")
        values.append(value)
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: target sum must be finite and positive")
    if abs(total - 1.0) > tolerance:
        # The dataset stores probabilities as decimal JSON values.  Renormalize
        # harmless decimal drift, while retaining a bounded guard against an
        # accidentally wrong schema or scale.
        if abs(total - 1.0) > 0.25:
            raise DecisionDataError(
                f"{workflow}/{case_id}/{question_id}: target sum {total:.8g} is not plausible"
            )
    normalized = tuple(value / total for value in values)
    if not math.isclose(math.fsum(normalized), 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: target failed normalization")
    return normalized


def normalize_case(
    row: Mapping[str, Any],
    workflow: str,
    *,
    partition: str,
    source_index: int | None = None,
    allow_official_test: bool = False,
    target_tolerance: float = DEFAULT_TARGET_TOLERANCE,
) -> list[DecisionExample]:
    """Expand one five-question case into typed decision examples."""

    if partition == "test" and not allow_official_test:
        raise OfficialTestAccessError("Test cases require allow_official_test=True during the final stage")
    case_id = _case_id(row, workflow)
    state = _state_object(row, workflow, case_id)
    questions = _questions_object(row, workflow, case_id)
    gold = _gold_object(row, workflow, case_id)
    state_text = canonical_text(state)
    state_hash = _state_hash(state)
    row_hash = _row_content_hash(row, workflow, case_id)
    examples: list[DecisionExample] = []
    for question_id, raw_question in questions.items():
        question = dict(raw_question)
        primitive = str(question.get("type", question.get("primitive", "")))
        if primitive not in PRIMITIVES:
            raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: invalid primitive {primitive!r}")
        answer = gold.get(question_id)
        criteria = question.get("criteria")
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, Mapping):
            raise DecisionDataError(f"{workflow}/{case_id}/{question_id}: criteria must be an object")
        labels = _ordered_labels(primitive, criteria)
        if primitive == "noul" and not labels:
            labels = ["false", "true"]
        descriptions = _option_descriptions(question, labels)
        probabilities = _probability_mapping(
            primitive, answer, labels, workflow=workflow, case_id=case_id, question_id=str(question_id)
        )
        target = validate_soft_target(
            probabilities,
            labels,
            workflow=workflow,
            case_id=case_id,
            question_id=str(question_id),
            tolerance=target_tolerance,
        )
        score_values: tuple[float, ...] | None = None
        if primitive == "score":
            try:
                score_values = tuple(float(label) for label in labels)
            except ValueError:
                score_values = None
        gold_label = answer.get("label") if isinstance(answer, Mapping) else None
        examples.append(
            DecisionExample(
                workflow=workflow,
                case_id=case_id,
                question_id=str(question_id),
                primitive=primitive,
                instructions=str(question.get("instructions", question.get("instruction", ""))),
                state_text=state_text,
                options=tuple(
                    Option(label=label, description=descriptions[label], canonical_index=index)
                    for index, label in enumerate(labels)
                ),
                target=target,
                source_partition=partition,
                state_hash=state_hash,
                row_hash=row_hash,
                score_values=score_values,
                gold_label=str(gold_label) if gold_label is not None else None,
                source_index=source_index,
                provenance={
                    "dataset_name": DATASET_NAME,
                    "workflow": workflow,
                    "case_id": case_id,
                    "question_id": str(question_id),
                    "primitive": primitive,
                    "canonical_labels": list(labels),
                    "source_partition": partition,
                    "row_hash": row_hash,
                },
            )
        )
    return examples


def normalize_rows(
    rows: Iterable[Mapping[str, Any]],
    workflow: str,
    *,
    partition: str,
    allow_official_test: bool = False,
    expected_questions_per_case: int | None = 5,
    target_tolerance: float = DEFAULT_TARGET_TOLERANCE,
) -> list[DecisionExample]:
    if partition == "test" and not allow_official_test:
        raise OfficialTestAccessError("Test normalization requires explicit final-stage access")
    materialized = validate_workflow_rows(
        rows,
        workflow,
        split=partition,
        expected_cases=None,
        expected_questions_per_case=expected_questions_per_case,
    )
    examples: list[DecisionExample] = []
    for index, row in enumerate(materialized):
        examples.extend(
            normalize_case(
                row,
                workflow,
                partition=partition,
                source_index=index,
                allow_official_test=allow_official_test,
                target_tolerance=target_tolerance,
            )
        )
    return examples


def _identity_rows(rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]], split: str) -> dict[str, dict[str, Any]]:
    identities: dict[str, dict[str, Any]] = {}
    for workflow in WORKFLOWS:
        for row in rows_by_workflow.get(workflow, []):
            case_id = _case_id(row, workflow)
            key = f"{workflow}/{case_id}"
            identities[key] = {
                "workflow": workflow,
                "case_id": case_id,
                "state_hash": _state_hash(_state_object(row, workflow, case_id)),
                "row_hash": _row_content_hash(row, workflow, case_id),
                "split": split,
            }
    return identities


def validate_no_overlap(
    split_manifest: SplitManifest | Mapping[str, Any],
    *,
    official_test_rows: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
) -> None:
    manifest = split_manifest.to_dict() if isinstance(split_manifest, SplitManifest) else dict(split_manifest)
    assignments = manifest.get("assignments", {})
    seen_ids: dict[str, str] = {}
    seen_states: dict[str, str] = {}
    state_hashes = manifest.get("state_hashes", {})
    for workflow, partitions in assignments.items():
        for partition, ids in partitions.items():
            for case_id in ids:
                key = f"{workflow}/{case_id}"
                if key in seen_ids:
                    raise DecisionDataError(f"case id overlaps {seen_ids[key]} and {partition}: {key}")
                seen_ids[key] = partition
                state_hash = state_hashes.get(key)
                if state_hash is not None:
                    if state_hash in seen_states:
                        raise DecisionDataError(f"normalized state overlaps {seen_states[state_hash]} and {key}")
                    seen_states[state_hash] = key
    if official_test_rows is None:
        return
    test_identities = _identity_rows(official_test_rows, "test")
    for key, identity in test_identities.items():
        if key in seen_ids:
            raise DecisionDataError(f"official test case overlaps development case: {key}")
        state_hash = identity["state_hash"]
        if state_hash in seen_states:
            raise DecisionDataError(f"official test state overlaps development state: {key}")


def create_split_manifest(
    train_rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    dataset_name: str = DATASET_NAME,
    dataset_revision: str = DATASET_REVISION,
    case_counts: Mapping[str, int] = DEFAULT_CASE_COUNTS,
    expected_train_cases: int | None = 300,
    expected_questions_per_case: int | None = 5,
    official_test_rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    existing_manifest: Mapping[str, Any] | None = None,
) -> SplitManifest:
    """Create or verify deterministic 240/30/30 case assignments per workflow."""

    if set(case_counts) != set(PARTITIONS):
        raise DecisionDataError(f"case_counts must contain exactly {PARTITIONS}")
    if sum(case_counts.values()) != (expected_train_cases if expected_train_cases is not None else sum(case_counts.values())):
        raise DecisionDataError("case_counts must sum to the expected training cardinality")
    assignments: dict[str, dict[str, tuple[str, ...]]] = {}
    state_hashes: dict[str, str] = {}
    for workflow in WORKFLOWS:
        rows = validate_workflow_rows(
            train_rows_by_workflow.get(workflow, []),
            workflow,
            split="train",
            expected_cases=expected_train_cases,
            expected_questions_per_case=expected_questions_per_case,
        )
        keyed = []
        for row in rows:
            case_id = _case_id(row, workflow)
            keyed.append((stable_hash({"seed": split_seed, "workflow": workflow, "case_id": case_id}), case_id, row))
            state_hashes[f"{workflow}/{case_id}"] = _state_hash(_state_object(row, workflow, case_id))
        keyed.sort(key=lambda item: (item[0], item[1]))
        cursor = 0
        assignments[workflow] = {}
        for partition in PARTITIONS:
            count = int(case_counts[partition])
            assignments[workflow][partition] = tuple(item[1] for item in keyed[cursor : cursor + count])
            cursor += count
    official_hashes: dict[str, str] = {}
    if official_test_rows_by_workflow is not None:
        for workflow, row in ((workflow, row) for workflow in WORKFLOWS for row in official_test_rows_by_workflow.get(workflow, [])):
            case_id = _case_id(row, workflow)
            key = f"{workflow}/{case_id}"
            official_hashes[key] = _state_hash(_state_object(row, workflow, case_id))
    draft = {
        "schema_version": DATA_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "dataset_revision": dataset_revision,
        "split_seed": split_seed,
        "workflows": list(WORKFLOWS),
        "assignments": {
            workflow: {partition: list(ids) for partition, ids in parts.items()}
            for workflow, parts in assignments.items()
        },
        "state_hashes": dict(sorted(state_hashes.items())),
        "official_test_hashes": dict(sorted(official_hashes.items())),
    }
    split_hash = stable_hash(draft)
    manifest = SplitManifest(
        schema_version=DATA_SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        split_seed=split_seed,
        workflows=WORKFLOWS,
        assignments=assignments,
        state_hashes=state_hashes,
        official_test_hashes=official_hashes,
        split_hash=split_hash,
    )
    validate_no_overlap(manifest, official_test_rows=official_test_rows_by_workflow)
    if existing_manifest is not None:
        existing_hash = existing_manifest.get("split_hash")
        if existing_hash != manifest.to_dict()["split_hash"]:
            raise ManifestMismatchError(
                f"persisted split hash {existing_hash!r} does not match prepared split {manifest.to_dict()['split_hash']!r}"
            )
    return manifest


def split_rows_by_manifest(
    rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]],
    split_manifest: SplitManifest | Mapping[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Materialize rows according to a previously persisted case assignment."""

    manifest = split_manifest.to_dict() if isinstance(split_manifest, SplitManifest) else split_manifest
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for workflow in WORKFLOWS:
        by_id = {_case_id(row, workflow): dict(row) for row in rows_by_workflow.get(workflow, [])}
        output[workflow] = {}
        for partition in PARTITIONS:
            ids = manifest.get("assignments", {}).get(workflow, {}).get(partition, [])
            missing = [case_id for case_id in ids if case_id not in by_id]
            if missing:
                raise ManifestMismatchError(f"{workflow}/{partition}: missing persisted case ids {missing[:5]}")
            output[workflow][partition] = [by_id[case_id] for case_id in ids]
    return output


def _encoder_encode(encoder: Any, text: str) -> list[int]:
    try:
        values = encoder.encode(text, allowed_special="all")
    except TypeError:
        values = encoder.encode(text)
    return [int(value) for value in values]


def _encoder_decode(encoder: Any, token_ids: Sequence[int]) -> str:
    decoder = getattr(encoder, "decode", None)
    if decoder is None:
        return ""
    return str(decoder(list(token_ids)))


def get_gpt2_encoder() -> Any:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("GPT-2 serialization requires tiktoken") from exc
    return tiktoken.get_encoding(TOKENIZER_NAME)


def _head_tail(values: Sequence[int], keep: int) -> tuple[list[int], list[int]]:
    if keep >= len(values):
        return list(values), []
    head = (keep + 1) // 2
    tail = keep - head
    return list(values[:head]), list(values[len(values) - tail :]) if tail else []


def serialize_decision(
    example: DecisionExample,
    *,
    encoder: Any | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    option_order: Sequence[int] | None = None,
) -> SerializedDecision:
    """Serialize a decision and truncate only the state token sequence."""

    if context_length < 1:
        raise DecisionDataError("context_length must be positive")
    encoder = encoder or get_gpt2_encoder()
    count = len(example.options)
    if option_order is None:
        order = tuple(range(count))
    else:
        order = tuple(int(index) for index in option_order)
        if sorted(order) != list(range(count)):
            raise DecisionDataError(f"option_order must be a permutation of 0..{count - 1}")
    options = tuple(example.options[index] for index in order)
    target = tuple(example.target[index] for index in order)
    state_intro_ids = _encoder_encode(encoder, "State:\n")
    state_ids = _encoder_encode(encoder, example.state_text)
    question_ids = _encoder_encode(encoder, f"\n\nQuestion: {example.instructions}\nOptions:\n")
    option_token_ids = tuple(
        tuple(_encoder_encode(encoder, f"[{option.label}] {option.description}\n")) for option in options
    )
    decision_ids = _encoder_encode(encoder, "Decision:")
    mandatory = len(state_intro_ids) + len(question_ids) + len(decision_ids) + sum(len(ids) for ids in option_token_ids)
    if mandatory > context_length:
        raise DecisionDataError(
            f"{example.workflow}/{example.case_id}/{example.question_id}: question/options require {mandatory} tokens, exceeds context {context_length}"
        )
    room = context_length - mandatory
    retained_head, retained_tail = _head_tail(state_ids, room)
    retained_state_ids = retained_head + retained_tail
    truncated = len(retained_state_ids) < len(state_ids)
    token_ids = list(state_intro_ids) + retained_state_ids + list(question_ids)
    question_start = len(state_intro_ids) + len(retained_state_ids)
    question_span = (question_start, question_start + len(question_ids))
    option_spans: list[tuple[int, int]] = []
    for option_ids in option_token_ids:
        start = len(token_ids)
        token_ids.extend(option_ids)
        option_spans.append((start, len(token_ids)))
    decision_start = len(token_ids)
    token_ids.extend(decision_ids)
    decision_span = (decision_start, len(token_ids))
    if not decision_ids:
        raise DecisionDataError("decision marker produced no tokens")
    provenance = dict(example.provenance)
    provenance.update(
        {
            "serialization_version": SERIALIZATION_VERSION,
            "tokenizer": TOKENIZER_NAME,
            "context_length": context_length,
            "option_order": list(order),
            "canonical_option_labels": list(example.labels),
            "serialized_option_labels": [option.label for option in options],
            "state_original_tokens": len(state_ids),
            "state_retained_tokens": len(retained_state_ids),
            "state_truncated": truncated,
            "retained_state_text": _encoder_decode(encoder, retained_state_ids),
            "canonical_text": (
                "State:\n"
                + _encoder_decode(encoder, retained_state_ids)
                + f"\n\nQuestion: {example.instructions}\nOptions:\n"
                + "".join(f"[{option.label}] {option.description}\n" for option in options)
                + "Decision:"
            ),
        }
    )
    provenance["serialized_hash"] = stable_hash({"token_ids": token_ids, "provenance": provenance})
    return SerializedDecision(
        token_ids=tuple(token_ids),
        decision_position=len(token_ids) - 1,
        decision_span=decision_span,
        question_span=question_span,
        option_spans=tuple(option_spans),
        target=tuple(target),
        labels=tuple(option.label for option in options),
        options=options,
        provenance=provenance,
        option_token_ids=option_token_ids,
        score_values=(
            tuple(example.score_values[index] for index in order)
            if example.score_values is not None else None
        ),
        state_text=_encoder_decode(encoder, retained_state_ids),
        instructions=example.instructions,
    )


def collate_decisions(
    batch: Sequence[SerializedDecision],
    *,
    pad_token_id: int = 50256,
) -> dict[str, Any]:
    """Pad sequence and candidate dimensions while retaining span metadata."""

    if not batch:
        raise DecisionDataError("cannot collate an empty batch")
    max_length = max(len(item.token_ids) for item in batch)
    max_options = max(item.option_count for item in batch)
    input_ids = [list(item.token_ids) + [pad_token_id] * (max_length - len(item.token_ids)) for item in batch]
    attention_mask = [[1] * len(item.token_ids) + [0] * (max_length - len(item.token_ids)) for item in batch]
    option_spans: list[list[list[int]]] = []
    option_mask: list[list[int]] = []
    targets: list[list[float]] = []
    labels: list[list[str | None]] = []
    for item in batch:
        option_spans.append([list(span) for span in item.option_spans] + [[-1, -1]] * (max_options - item.option_count))
        option_mask.append([1] * item.option_count + [0] * (max_options - item.option_count))
        targets.append(list(item.target) + [0.0] * (max_options - item.option_count))
        labels.append(list(item.labels) + [None] * (max_options - item.option_count))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": [len(item.token_ids) for item in batch],
        "decision_positions": [item.decision_position for item in batch],
        "option_spans": option_spans,
        "option_mask": option_mask,
        "targets": targets,
        "labels": labels,
        "provenance": [dict(item.provenance) for item in batch],
    }


def permute_serialized(serialized: SerializedDecision, permutation: Sequence[int]) -> SerializedDecision:
    """Reorder contiguous candidate spans and remap targets to the new order."""

    count = serialized.option_count
    order = tuple(int(index) for index in permutation)
    if sorted(order) != list(range(count)):
        raise DecisionDataError(f"permutation must contain each candidate index exactly once")
    spans = serialized.option_spans
    if spans and any(spans[index][1] != spans[index + 1][0] for index in range(count - 1)):
        raise DecisionDataError("candidate spans must be contiguous to permute a serialized decision")
    first = spans[0][0]
    last = spans[-1][1]
    chunks = [list(serialized.token_ids[start:end]) for start, end in spans]
    reordered = list(serialized.token_ids[:first])
    new_spans: list[tuple[int, int]] = []
    for index in order:
        start = len(reordered)
        reordered.extend(chunks[index])
        new_spans.append((start, len(reordered)))
    reordered.extend(serialized.token_ids[last:])
    provenance = dict(serialized.provenance)
    provenance["permutation"] = list(order)
    provenance["option_order"] = list(order)
    return SerializedDecision(
        token_ids=tuple(reordered),
        decision_position=serialized.decision_position,
        decision_span=serialized.decision_span,
        question_span=serialized.question_span,
        option_spans=tuple(new_spans),
        target=tuple(serialized.target[index] for index in order),
        labels=tuple(serialized.labels[index] for index in order),
        options=tuple(serialized.options[index] for index in order),
        provenance=provenance,
        option_token_ids=tuple(serialized.option_token_ids[index] for index in order),
        score_values=(
            tuple(serialized.score_values[index] for index in order)
            if serialized.score_values is not None else None
        ),
        state_text=serialized.state_text,
        instructions=serialized.instructions,
    )


def deterministic_option_permutation(
    example: DecisionExample | SerializedDecision,
    *,
    seed: int,
    permutation_index: int = 0,
) -> tuple[int, ...]:
    count = len(example.options) if isinstance(example, SerializedDecision) else example.option_count
    generator = random.Random(stable_hash({"seed": seed, "permutation": permutation_index, "labels": list(example.labels)}))
    order = list(range(count))
    generator.shuffle(order)
    return tuple(order)


def _percentile(values: Sequence[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _length_summary(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p90": None, "p99": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
    }


def build_data_manifest(
    examples: Iterable[DecisionExample | SerializedDecision],
    *,
    split_manifest: SplitManifest | Mapping[str, Any] | None = None,
    dataset_name: str = DATASET_NAME,
    dataset_revision: str = DATASET_REVISION,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    tokenizer_name: str = TOKENIZER_NAME,
    serialization_version: str = SERIALIZATION_VERSION,
    training_fit_gate: float = 0.99,
    extra_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the immutable audit/manifest payload consumed downstream."""

    items = list(examples)
    if not items:
        raise DecisionDataError("cannot build a data manifest from no examples")
    split_counts: Counter[str] = Counter()
    workflow_counts: Counter[str] = Counter()
    primitive_counts: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    lengths: list[int] = []
    state_lengths: list[int] = []
    truncated = 0
    content_hashes: list[str] = []
    for item in items:
        provenance = dict(item.provenance)
        partition = str(provenance.get("source_partition", "unknown"))
        workflow = str(provenance.get("workflow", "unknown"))
        primitive = str(provenance.get("primitive", "unknown"))
        split_counts[partition] += 1
        workflow_counts[workflow] += 1
        primitive_counts[primitive] += 1
        option_counts[str(item.option_count)] += 1
        lengths.append(len(item.token_ids))
        state_lengths.append(int(provenance.get("state_original_tokens", 0)))
        truncated += int(bool(provenance.get("state_truncated", False)))
        content_hashes.append(
            stable_hash(item.to_dict() if hasattr(item, "to_dict") else item)
        )
    train_count = split_counts.get("train", 0)
    train_truncation_rate = (
        sum(
            int(bool(item.provenance.get("state_truncated", False)))
            for item in items
            if item.provenance.get("source_partition") == "train"
        )
        / train_count
        if train_count
        else None
    )
    payload: dict[str, Any] = {
        "schema_version": DATA_SCHEMA_VERSION,
        "dataset": {
            "name": dataset_name,
            "revision": dataset_revision,
        },
        "tokenizer": {"name": tokenizer_name, "serialization_version": serialization_version},
        "context_length": context_length,
        "split_manifest": split_manifest.to_dict() if isinstance(split_manifest, SplitManifest) else _jsonable(split_manifest),
        "counts": {
            "examples": len(items),
            "by_partition": dict(sorted(split_counts.items())),
            "by_workflow": dict(sorted(workflow_counts.items())),
            "by_primitive": dict(sorted(primitive_counts.items())),
            "by_option_count": dict(sorted(option_counts.items(), key=lambda pair: int(pair[0]))),
        },
        "lengths": {
            "serialized_tokens": _length_summary(lengths),
            "original_state_tokens": _length_summary(state_lengths),
        },
        "truncation": {
            "count": truncated,
            "rate": truncated / len(items),
            "training_count": train_count,
            "training_rate": train_truncation_rate,
            "fit_gate": training_fit_gate,
            "training_fit_gate_passed": train_truncation_rate is not None and (1.0 - train_truncation_rate) >= training_fit_gate,
        },
        "content_hash": stable_hash(sorted(content_hashes)),
        "example_hashes": sorted(content_hashes),
    }
    if extra_identity:
        payload["extra_identity"] = _jsonable(dict(extra_identity))
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def write_data_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    expected = stable_hash({key: value for key, value in manifest.items() if key != "manifest_hash"})
    recorded = manifest.get("manifest_hash")
    if recorded is not None and recorded != expected:
        raise ManifestMismatchError("data manifest hash does not match its canonical contents")
    atomic_write_json(path, dict(manifest))


def audit_serialized_examples(
    examples: Iterable[DecisionExample],
    *,
    encoder: Any | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    split_manifest: SplitManifest | Mapping[str, Any] | None = None,
) -> tuple[list[SerializedDecision], dict[str, Any]]:
    """Serialize examples and return both rows and their manifest."""

    serialized = [serialize_decision(example, encoder=encoder, context_length=context_length) for example in examples]
    return serialized, build_data_manifest(
        serialized,
        split_manifest=split_manifest,
        context_length=context_length,
    )


def decision_to_dict(item: SerializedDecision, *, tensor_backend: str = "torch") -> dict[str, Any]:
    """Convert one serialized decision to the trainer's stable dictionary API.

    The default backend returns CPU ``torch.Tensor`` objects.  Keeping this
    conversion at the boundary means the data preparation logic remains
    usable by audit/reporting code that only needs Python values.
    """

    if tensor_backend != "torch":
        raise DecisionDataError(f"unsupported tensor backend {tensor_backend!r}; only 'torch' is supported")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("decision_to_dict requires PyTorch for Tensor outputs") from exc
    provenance = dict(item.provenance)
    canonical_labels = provenance.get("canonical_labels")
    if canonical_labels is None:
        canonical_labels = provenance.get("canonical_option_labels", list(item.labels))
    return {
        "input_ids": torch.tensor(item.token_ids, dtype=torch.long),
        "decision_index": int(item.decision_position),
        "option_spans": [tuple(map(int, span)) for span in item.option_spans],
        "target": torch.tensor(item.target, dtype=torch.float32),
        "primitive": str(provenance.get("primitive", "")),
        "workflow": str(provenance.get("workflow", "")),
        "case_id": str(provenance.get("case_id", "")),
        "question_id": str(provenance.get("question_id", "")),
        "truncated": bool(provenance.get("state_truncated", False)),
        "original_length": int(provenance.get("state_original_tokens", len(item.token_ids))),
        "retained_length": int(provenance.get("state_retained_tokens", len(item.token_ids))),
        "score_values": list(item.score_values) if item.score_values is not None else None,
        "state_text": item.state_text,
        "instructions": item.instructions,
        "options": [option.to_dict() for option in item.options],
        "canonical_labels": [str(label) for label in canonical_labels],
        "serialized_labels": list(item.labels),
        "option_order": list(provenance.get("option_order", range(item.option_count))),
        "option_count": int(item.option_count),
        "provenance": provenance,
    }


def prepare_example_dict(
    example: DecisionExample | SerializedDecision,
    *,
    encoder: Any | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    option_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Prepare one trainer-facing example dictionary.

    A raw ``DecisionExample`` is serialized here.  A ``SerializedDecision``
    is reused as-is unless an option order is explicitly requested.
    """

    if isinstance(example, DecisionExample):
        serialized = serialize_decision(
            example,
            encoder=encoder,
            context_length=context_length,
            option_order=option_order,
        )
    elif isinstance(example, SerializedDecision):
        if option_order is not None:
            serialized = permute_serialized(example, option_order)
        else:
            serialized = example
    else:
        raise TypeError(f"expected DecisionExample or SerializedDecision, got {type(example).__name__}")
    return decision_to_dict(serialized)


def prepare_examples(
    examples: Iterable[DecisionExample | SerializedDecision],
    *,
    encoder: Any | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
) -> list[dict[str, Any]]:
    """Prepare a sequence of raw or serialized examples for a trainer."""

    return [
        prepare_example_dict(example, encoder=encoder, context_length=context_length)
        for example in examples
    ]


def _serialize_partition_examples(
    rows: Iterable[Mapping[str, Any]],
    workflow: str,
    *,
    partition: str,
    config: DataConfig,
    encoder: Any | None,
    allow_official_test: bool = False,
) -> list[dict[str, Any]]:
    normalized = normalize_rows(
        rows,
        workflow,
        partition=partition,
        allow_official_test=allow_official_test,
        expected_questions_per_case=config.expected_questions_per_case,
        target_tolerance=config.target_tolerance,
    )
    return [
        prepare_example_dict(example, encoder=encoder, context_length=config.context_length)
        for example in normalized
    ]


def load_decision_splits(
    train_rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    *,
    test_rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    config: DataConfig = DataConfig(),
    encoder: Any | None = None,
    include_test: bool = False,
    stage: str = "development",
    loader: Callable[..., Any] | None = None,
    existing_manifest: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return trainer-ready ``train/validation/calibration/test`` lists.

    Development rows are always sourced from the training split and assigned
    by case.  ``test`` is returned as an empty list by default.  To populate
    it, callers must set ``include_test=True`` and ``stage='final'``; when
    rows are not supplied, that explicit path is the only path that invokes
    the official test loader.
    """

    if train_rows_by_workflow is None:
        train_rows_by_workflow = load_all_training_workflows(
            revision=config.dataset_revision,
            dataset_name=config.dataset_name,
            loader=loader,
            expected_cases=config.expected_train_cases,
            expected_questions_per_case=config.expected_questions_per_case,
        )
    split_manifest = create_split_manifest(
        train_rows_by_workflow,
        split_seed=config.split_seed,
        dataset_name=config.dataset_name,
        dataset_revision=config.dataset_revision,
        case_counts=config.case_counts,
        expected_train_cases=config.expected_train_cases,
        expected_questions_per_case=config.expected_questions_per_case,
        existing_manifest=existing_manifest,
    )
    split_rows = split_rows_by_manifest(train_rows_by_workflow, split_manifest)
    result: dict[str, list[dict[str, Any]]] = {partition: [] for partition in (*PARTITIONS, "test")}
    for partition in PARTITIONS:
        result[partition] = [
            item
            for workflow in WORKFLOWS
            for item in _serialize_partition_examples(
                split_rows[workflow][partition],
                workflow,
                partition=partition,
                config=config,
                encoder=encoder,
            )
        ]
    if include_test:
        if stage != "final":
            raise OfficialTestAccessError("include_test=True requires stage='final'")
        if test_rows_by_workflow is None:
            test_rows_by_workflow = load_official_test_workflows(
                revision=config.dataset_revision,
                dataset_name=config.dataset_name,
                loader=loader,
                expected_cases=config.expected_test_cases,
                expected_questions_per_case=config.expected_questions_per_case,
            )
        else:
            test_rows_by_workflow = {
                workflow: validate_workflow_rows(
                    test_rows_by_workflow.get(workflow, []),
                    workflow,
                    split="test",
                    expected_cases=config.expected_test_cases,
                    expected_questions_per_case=config.expected_questions_per_case,
                )
                for workflow in WORKFLOWS
            }
        validate_no_overlap(split_manifest, official_test_rows=test_rows_by_workflow)
        result["test"] = [
            item
            for workflow in WORKFLOWS
            for item in _serialize_partition_examples(
                test_rows_by_workflow.get(workflow, []),
                workflow,
                partition="test",
                config=config,
                encoder=encoder,
                allow_official_test=True,
            )
        ]
    return result


def prepare_development_data(
    rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    config: DataConfig = DataConfig(),
    encoder: Any | None = None,
    existing_manifest: Mapping[str, Any] | None = None,
) -> tuple[SplitManifest, dict[str, list[SerializedDecision]], dict[str, Any]]:
    """Split, normalize, serialize, and audit the three development partitions."""

    split_manifest = create_split_manifest(
        rows_by_workflow,
        split_seed=config.split_seed,
        dataset_name=config.dataset_name,
        dataset_revision=config.dataset_revision,
        case_counts=config.case_counts,
        expected_train_cases=config.expected_train_cases,
        expected_questions_per_case=config.expected_questions_per_case,
        existing_manifest=existing_manifest,
    )
    split_rows = split_rows_by_manifest(rows_by_workflow, split_manifest)
    all_serialized: dict[str, list[SerializedDecision]] = {}
    all_items: list[SerializedDecision] = []
    for partition in PARTITIONS:
        normalized = [
            example
            for workflow in WORKFLOWS
            for example in normalize_rows(
                split_rows[workflow][partition],
                workflow,
                partition=partition,
                expected_questions_per_case=config.expected_questions_per_case,
                target_tolerance=config.target_tolerance,
            )
        ]
        serialized = [serialize_decision(example, encoder=encoder, context_length=config.context_length) for example in normalized]
        all_serialized[partition] = serialized
        all_items.extend(serialized)
    manifest = build_data_manifest(
        all_items,
        split_manifest=split_manifest,
        dataset_name=config.dataset_name,
        dataset_revision=config.dataset_revision,
        context_length=config.context_length,
        tokenizer_name=config.tokenizer_name,
    )
    if not manifest["truncation"]["training_fit_gate_passed"]:
        raise DecisionDataError(
            "512-token training fit gate failed; qualify a larger context before production training"
        )
    return split_manifest, all_serialized, manifest


def prepare_official_test_data(
    rows_by_workflow: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    config: DataConfig = DataConfig(),
    encoder: Any | None = None,
    split_manifest: SplitManifest | Mapping[str, Any] | None = None,
) -> tuple[list[SerializedDecision], dict[str, Any]]:
    """Explicit final-stage test preparation; callers must opt in by function."""

    if split_manifest is None:
        raise OfficialTestAccessError("official test preparation requires the locked development split manifest")
    examples = [
        example
        for workflow in WORKFLOWS
        for example in normalize_rows(
            rows_by_workflow.get(workflow, []),
            workflow,
            partition="test",
            allow_official_test=True,
            expected_questions_per_case=config.expected_questions_per_case,
            target_tolerance=config.target_tolerance,
        )
    ]
    serialized = [serialize_decision(example, encoder=encoder, context_length=config.context_length) for example in examples]
    manifest = build_data_manifest(
        serialized,
        split_manifest=split_manifest,
        dataset_name=config.dataset_name,
        dataset_revision=config.dataset_revision,
        context_length=config.context_length,
        tokenizer_name=config.tokenizer_name,
        extra_identity={"official_test": True},
    )
    return serialized, manifest


# Backward-friendly aliases for callers that prefer shorter names.
load_training_workflows = load_all_training_workflows
load_test_workflows = load_official_test_workflows
normalize_decision_rows = normalize_rows
serialize_example = serialize_decision
make_split_manifest = create_split_manifest
collate = collate_decisions
prepare_example = prepare_example_dict
load_splits = load_decision_splits


__all__ = [
    "DATASET_NAME",
    "DATASET_REVISION",
    "DATA_SCHEMA_VERSION",
    "SERIALIZATION_VERSION",
    "TOKENIZER_NAME",
    "WORKFLOWS",
    "PARTITIONS",
    "PRIMITIVES",
    "DecisionDataError",
    "OfficialTestAccessError",
    "ManifestMismatchError",
    "Option",
    "DecisionExample",
    "SerializedDecision",
    "SplitManifest",
    "DataConfig",
    "canonical_json",
    "stable_hash",
    "sha256_file",
    "atomic_write_json",
    "load_workflow_rows",
    "load_all_training_workflows",
    "load_official_test_workflows",
    "load_training_workflows",
    "load_test_workflows",
    "validate_workflow_rows",
    "validate_soft_target",
    "normalize_case",
    "normalize_rows",
    "normalize_decision_rows",
    "validate_no_overlap",
    "create_split_manifest",
    "make_split_manifest",
    "split_rows_by_manifest",
    "get_gpt2_encoder",
    "serialize_decision",
    "serialize_example",
    "collate_decisions",
    "collate",
    "permute_serialized",
    "deterministic_option_permutation",
    "build_data_manifest",
    "write_data_manifest",
    "audit_serialized_examples",
    "decision_to_dict",
    "prepare_example",
    "prepare_example_dict",
    "prepare_examples",
    "load_decision_splits",
    "load_splits",
    "prepare_development_data",
    "prepare_official_test_data",
]
