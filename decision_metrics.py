"""Metrics and uncertainty utilities for typed decision experiments.

The public functions use ``predictions`` first and ``targets`` second.  A
distribution may be a mapping from option labels to numbers, a one-dimensional
list/NumPy array/torch tensor, or a batch of those values.  Mapping keys are
used to align rows; positional order is used for array-like inputs.

All probability inputs are checked rather than silently renormalized.  Values
must be finite, non-negative, and sum to one within ``atol``.  This catches
corrupt prediction rows before they enter a report.

Metric conventions
------------------
* Soft NLL is ``mean_i[-sum_k y[i,k] log(max(p[i,k], eps))]``.
* Top-1 accuracy compares ``argmax(p)`` with ``argmax(y)``.  NumPy's first
  index wins ties.
* Soft accuracy is ``mean_i[sum_k p[i,k] y[i,k]]``.
* Multiclass Brier is ``mean_i[sum_k (p[i,k] - y[i,k])**2]``.
* ECE bins confidence with equally spaced intervals ``[a, b)`` and makes the
  final interval ``[a, 1]``.  Confidence is the largest predicted
  probability, while correctness is the target probability assigned to the
  predicted class.  This reduces to ordinary correctness for one-hot labels
  and remains meaningful for soft teacher distributions.
* Score expected-value MAE compares ``sum_k p[k] * score_values[k]`` with the
  corresponding target expectation, then averages absolute errors.

Case bootstrap functions resample complete cases with replacement.  All rows
belonging to a sampled case are kept together, preserving the dependence among
the five decisions from one Typed Decisions case.  Intervals use percentile
quantiles with NumPy's default linear interpolation.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, TypeAlias

import numpy as np


NumericInput: TypeAlias = Any
Metric: TypeAlias = Callable[[np.ndarray, np.ndarray], float]

_DEFAULT_ATOL = 1e-6
_DEFAULT_EPS = 1e-12


@dataclass(frozen=True)
class CalibrationBin:
    """Summary for one ECE confidence bin."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    mean_correctness: float
    absolute_gap: float


@dataclass(frozen=True)
class TemperatureFitResult:
    """Result of positive temperature scaling on held-out logits."""

    temperature: float
    initial_nll: float
    final_nll: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile interval for a statistic estimated by case bootstrap."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    n_cases: int
    seed: int | None


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Intervals for two model statistics and their paired difference.

    ``difference`` is always ``left - right`` for every bootstrap replicate.
    """

    left: BootstrapInterval
    right: BootstrapInterval
    difference: BootstrapInterval


def _is_tensor(value: Any) -> bool:
    """Return whether ``value`` looks like a torch tensor without importing torch."""

    return hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "shape")


def _to_numpy(value: Any, *, name: str) -> np.ndarray:
    """Convert NumPy, torch, and ordinary numeric values to float64 arrays."""

    if _is_tensor(value):
        try:
            value = value.detach().cpu().numpy()
        except Exception as exc:  # pragma: no cover - depends on tensor backend
            raise TypeError(f"{name} could not be converted from a tensor") from exc
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc


def _mapping_labels(value: Mapping[Hashable, Any]) -> tuple[Hashable, ...]:
    labels = tuple(value.keys())
    if not labels:
        raise ValueError("a probability mapping must contain at least one option")
    if len(set(labels)) != len(labels):  # Defensive for unusual Mapping types.
        raise ValueError("probability mapping contains duplicate option labels")
    return labels


def _coerce_matrix(
    value: NumericInput,
    *,
    name: str,
    labels: Sequence[Hashable] | None = None,
) -> tuple[np.ndarray, tuple[Hashable, ...] | None]:
    """Convert a distribution or batch of distributions to an ``(N, K)`` array."""

    if isinstance(value, Mapping):
        row_labels = _mapping_labels(value)
        if labels is not None and tuple(labels) != row_labels:
            raise ValueError(f"{name} option labels do not match the other distribution")
        arr = np.asarray([value[label] for label in row_labels], dtype=np.float64)[None, :]
        return arr, row_labels

    if _is_tensor(value) or isinstance(value, np.ndarray):
        arr = _to_numpy(value, name=name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if not items:
            raise ValueError(f"{name} must not be empty")
        if all(isinstance(item, Mapping) for item in items):
            first_labels = _mapping_labels(items[0])
            if labels is not None and tuple(labels) != first_labels:
                raise ValueError(f"{name} option labels do not match the other distribution")
            rows = []
            for index, item in enumerate(items):
                item_labels = _mapping_labels(item)
                if item_labels != first_labels and set(item_labels) != set(first_labels):
                    raise ValueError(f"{name}[{index}] option labels do not match the first row")
                rows.append([item[label] for label in first_labels])
            arr = np.asarray(rows, dtype=np.float64)
            return arr, first_labels
        if any(isinstance(item, Mapping) for item in items):
            raise TypeError(f"{name} cannot mix mappings and positional rows")
        # Handling tensor rows explicitly also supports list[torch.Tensor].
        if any(_is_tensor(item) for item in items):
            arr = np.stack([_to_numpy(item, name=f"{name}[{i}]") for i, item in enumerate(items)])
        else:
            arr = _to_numpy(items, name=name)
    else:
        arr = _to_numpy(value, name=name)

    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one distribution or a 2-D batch; got shape {arr.shape}")
    if labels is not None and arr.shape[1] != len(labels):
        raise ValueError(f"{name} has {arr.shape[1]} options but {len(labels)} labels were supplied")
    return arr, tuple(labels) if labels is not None else None


def _coerce_numeric_matrix(value: NumericInput, *, name: str) -> np.ndarray:
    """Convert numeric vectors/matrices, allowing mappings but not probabilities."""

    arr, _ = _coerce_matrix(value, name=name)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _coerce_bootstrap_rows(value: NumericInput, *, name: str) -> np.ndarray:
    """Convert bootstrap observations, treating a 1-D input as one value per row."""

    if isinstance(value, Mapping):
        arr = _to_numpy(list(value.values()), name=name)
    elif _is_tensor(value) or isinstance(value, np.ndarray):
        arr = _to_numpy(value, name=name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if not items:
            raise ValueError(f"{name} must not be empty")
        if any(_is_tensor(item) for item in items):
            arr = np.stack([_to_numpy(item, name=f"{name}[{i}]") for i, item in enumerate(items)])
        else:
            arr = _to_numpy(items, name=name)
    else:
        arr = _to_numpy(value, name=name)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must contain one or more 1-D/2-D observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def validate_probability_distributions(
    probabilities: NumericInput,
    *,
    name: str = "probabilities",
    atol: float = _DEFAULT_ATOL,
) -> np.ndarray:
    """Validate and return probability distributions as an ``(N, K)`` array.

    The function intentionally does not normalize inputs.  A row whose sum is
    outside ``1 +/- atol`` is rejected so malformed prediction artifacts cannot
    be hidden by a later metric calculation.
    """

    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and non-negative")
    arr, _ = _coerce_matrix(probabilities, name=name)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one option")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    if np.any(arr < 0):
        raise ValueError(f"{name} contains negative probabilities")
    sums = np.sum(arr, axis=1)
    if not np.all(np.isclose(sums, 1.0, rtol=0.0, atol=atol)):
        bad = int(np.flatnonzero(~np.isclose(sums, 1.0, rtol=0.0, atol=atol))[0])
        raise ValueError(f"{name}[{bad}] sums to {sums[bad]:.9g}, expected 1 within atol={atol}")
    return arr


# Short aliases make the validation utility convenient at call sites.
validate_probabilities = validate_probability_distributions


def _validated_pair(
    predictions: NumericInput,
    targets: NumericInput,
    *,
    atol: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred, labels = _coerce_matrix(predictions, name="predictions")
    target, _ = _coerce_matrix(targets, name="targets", labels=labels)
    pred = validate_probability_distributions(pred, name="predictions", atol=atol)
    target = validate_probability_distributions(target, name="targets", atol=atol)
    if pred.shape != target.shape:
        raise ValueError(f"predictions and targets must have the same shape; got {pred.shape} and {target.shape}")
    return pred, target


def _reduce(values: np.ndarray, reduction: str) -> float | np.ndarray:
    if reduction == "none":
        return values
    if reduction == "mean":
        return float(np.mean(values))
    if reduction == "sum":
        return float(np.sum(values))
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def soft_nll(
    predictions: NumericInput,
    targets: NumericInput,
    *,
    reduction: str = "mean",
    eps: float = _DEFAULT_EPS,
    atol: float = _DEFAULT_ATOL,
) -> float | np.ndarray:
    """Compute soft-label cross-entropy from validated probabilities."""

    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    pred, target = _validated_pair(predictions, targets, atol=atol)
    per_row = -np.sum(target * np.log(np.clip(pred, eps, 1.0)), axis=1)
    return _reduce(per_row, reduction)


def top1_accuracy(
    predictions: NumericInput,
    targets: NumericInput,
    *,
    atol: float = _DEFAULT_ATOL,
) -> float:
    """Return hard top-1 accuracy using the target distribution's argmax."""

    pred, target = _validated_pair(predictions, targets, atol=atol)
    return float(np.mean(np.argmax(pred, axis=1) == np.argmax(target, axis=1)))


def soft_accuracy(
    predictions: NumericInput,
    targets: NumericInput,
    *,
    reduction: str = "mean",
    atol: float = _DEFAULT_ATOL,
) -> float | np.ndarray:
    """Return probability mass assigned to the soft gold distribution."""

    pred, target = _validated_pair(predictions, targets, atol=atol)
    return _reduce(np.sum(pred * target, axis=1), reduction)


def multiclass_brier(
    predictions: NumericInput,
    targets: NumericInput,
    *,
    reduction: str = "mean",
    atol: float = _DEFAULT_ATOL,
) -> float | np.ndarray:
    """Compute the multiclass Brier score, with lower values being better."""

    pred, target = _validated_pair(predictions, targets, atol=atol)
    return _reduce(np.sum((pred - target) ** 2, axis=1), reduction)


def calibration_bins(
    predictions: NumericInput,
    targets: NumericInput,
    *,
    n_bins: int = 15,
    atol: float = _DEFAULT_ATOL,
) -> tuple[CalibrationBin, ...]:
    """Return ECE bin summaries using equal-width confidence intervals.

    Bins are ``[j/n_bins, (j+1)/n_bins)`` except the last, which includes its
    right endpoint.  Empty bins are returned with ``count=0`` and NaN summary
    values so callers can preserve the fixed bin layout in reports.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    pred, target = _validated_pair(predictions, targets, atol=atol)
    predicted_class = np.argmax(pred, axis=1)
    confidence = np.max(pred, axis=1)
    correctness = target[np.arange(pred.shape[0]), predicted_class]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(np.floor(confidence * n_bins).astype(int), n_bins - 1)
    result: list[CalibrationBin] = []
    for index in range(n_bins):
        selected = bin_indices == index
        lower, upper = float(edges[index]), float(edges[index + 1])
        count = int(np.sum(selected))
        if count:
            mean_confidence = float(np.mean(confidence[selected]))
            mean_correctness = float(np.mean(correctness[selected]))
            gap = abs(mean_confidence - mean_correctness)
        else:
            mean_confidence = float("nan")
            mean_correctness = float("nan")
            gap = float("nan")
        result.append(CalibrationBin(lower, upper, count, mean_confidence, mean_correctness, gap))
    return tuple(result)


def expected_calibration_error(
    predictions: NumericInput,
    targets: NumericInput,
    *,
    n_bins: int = 15,
    atol: float = _DEFAULT_ATOL,
) -> float:
    """Compute expected calibration error under the module's documented bins."""

    bins = calibration_bins(predictions, targets, n_bins=n_bins, atol=atol)
    total = sum(item.count for item in bins)
    return float(sum(item.count * item.absolute_gap for item in bins if item.count) / total)


ece = expected_calibration_error


def score_expected_value_mae(
    predictions: NumericInput,
    targets: NumericInput,
    score_values: Sequence[float] | np.ndarray,
    *,
    atol: float = _DEFAULT_ATOL,
) -> float:
    """Compute MAE between predicted and target expected ordered scores."""

    pred, target = _validated_pair(predictions, targets, atol=atol)
    scores = _to_numpy(score_values, name="score_values").reshape(-1)
    if scores.size != pred.shape[1]:
        raise ValueError(f"score_values has length {scores.size}, expected {pred.shape[1]}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_values contains non-finite values")
    return float(np.mean(np.abs(pred @ scores - target @ scores)))


def probabilities_from_logits(
    logits: NumericInput,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert finite logits to a stable 2-D probability matrix."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    values = _coerce_numeric_matrix(logits, name="logits") / temperature
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


def apply_temperature(logits: NumericInput, temperature: float) -> np.ndarray:
    """Alias for :func:`probabilities_from_logits` with an explicit temperature."""

    return probabilities_from_logits(logits, temperature=temperature)


def fit_temperature(
    logits: NumericInput,
    targets: NumericInput,
    *,
    initial_temperature: float = 1.0,
    max_iter: int = 100,
    learning_rate: float = 0.05,
    tolerance: float = 1e-8,
    atol: float = _DEFAULT_ATOL,
) -> TemperatureFitResult:
    """Fit one finite positive temperature using held-out soft targets.

    The optimization variable is ``log(T)``, so every returned temperature is
    positive.  PyTorch is imported lazily because the other metrics only need
    NumPy.  Logits and targets are copied to CPU float64 tensors; this utility
    is intended for calibration-sized arrays rather than model training.
    """

    if not math.isfinite(initial_temperature) or initial_temperature <= 0:
        raise ValueError("initial_temperature must be finite and positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project declares torch
        raise RuntimeError("fit_temperature requires PyTorch to be installed") from exc

    logit_array, logit_labels = _coerce_matrix(logits, name="logits")
    if logit_array.ndim != 2 or logit_array.shape[0] < 1 or logit_array.shape[1] < 1:
        raise ValueError("logits must contain at least one row and one option")
    if not np.all(np.isfinite(logit_array)):
        raise ValueError("logits contains non-finite values")
    target_array, _ = _coerce_matrix(targets, name="targets", labels=logit_labels)
    target_array = validate_probability_distributions(target_array, name="targets", atol=atol)
    if logit_array.shape != target_array.shape:
        raise ValueError(f"logits and targets must have the same shape; got {logit_array.shape} and {target_array.shape}")

    logit_tensor = torch.as_tensor(logit_array, dtype=torch.float64)
    target_tensor = torch.as_tensor(target_array, dtype=torch.float64)
    log_temperature = torch.tensor(math.log(initial_temperature), dtype=torch.float64, requires_grad=True)

    def loss() -> Any:
        # Clamp only the exponentiated parameter to keep pathological optimizer
        # steps finite while preserving strict positivity.
        temperature = torch.exp(log_temperature).clamp(min=1e-8, max=1e8)
        return -(target_tensor * torch.log_softmax(logit_tensor / temperature, dim=-1)).sum(dim=-1).mean()

    initial_nll = float(loss().detach().cpu().item())
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=learning_rate,
        max_iter=max_iter,
        tolerance_grad=tolerance,
        tolerance_change=tolerance,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Any:
        optimizer.zero_grad()
        objective = loss()
        objective.backward()
        return objective

    optimizer.step(closure)
    final_nll = float(loss().detach().cpu().item())
    temperature = float(torch.exp(log_temperature.detach()).clamp(min=1e-8, max=1e8).cpu().item())
    if not math.isfinite(temperature) or not math.isfinite(final_nll):
        raise FloatingPointError("temperature fitting produced a non-finite result")
    state = optimizer.state.get(log_temperature, {})
    iterations = int(state.get("n_iter", state.get("func_evals", max_iter)))
    converged = abs(initial_nll - final_nll) <= tolerance or iterations < max_iter
    return TemperatureFitResult(temperature, initial_nll, final_nll, iterations, converged)


fit_positive_temperature = fit_temperature


def _case_groups(case_ids: Sequence[Hashable]) -> tuple[tuple[Hashable, ...], tuple[np.ndarray, ...]]:
    ids = list(case_ids)
    if not ids:
        raise ValueError("case_ids must not be empty")
    groups: dict[Hashable, list[int]] = {}
    for index, case_id in enumerate(ids):
        try:
            groups.setdefault(case_id, []).append(index)
        except TypeError as exc:
            raise TypeError("case_ids must contain hashable identifiers") from exc
    labels = tuple(groups.keys())
    indices = tuple(np.asarray(groups[label], dtype=np.int64) for label in labels)
    return labels, indices


def _bootstrap_setup(
    case_ids: Sequence[Hashable],
    *,
    n_resamples: int,
    confidence_level: float,
    seed: int | None,
) -> tuple[tuple[Hashable, ...], tuple[np.ndarray, ...], np.random.Generator]:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    labels, groups = _case_groups(case_ids)
    return labels, groups, np.random.default_rng(seed)


def _resampled_indices(groups: tuple[np.ndarray, ...], sampled_case_indices: np.ndarray) -> np.ndarray:
    return np.concatenate([groups[int(case_index)] for case_index in sampled_case_indices])


def _interval(
    estimate: float,
    samples: np.ndarray,
    *,
    confidence_level: float,
    n_cases: int,
    n_resamples: int,
    seed: int | None,
) -> BootstrapInterval:
    if not math.isfinite(estimate) or not np.all(np.isfinite(samples)):
        raise ValueError("bootstrap statistic produced a non-finite value")
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapInterval(float(estimate), float(lower), float(upper), confidence_level, n_resamples, n_cases, seed)


def case_level_bootstrap(
    case_ids: Sequence[Hashable],
    values: NumericInput,
    *,
    statistic: Callable[[np.ndarray], float] = lambda sample: float(np.mean(sample)),
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = 2028,
) -> BootstrapInterval:
    """Bootstrap a statistic while resampling complete cases.

    ``values`` must have one leading row per ``case_ids`` entry.  The default
    statistic is the mean, so passing per-decision losses, correctness values,
    or Brier contributions gives a case-clustered interval.  A custom
    statistic may consume the resampled 2-D array for nonlinear metrics.
    """

    labels, groups, rng = _bootstrap_setup(
        case_ids, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed
    )
    data = _coerce_bootstrap_rows(values, name="values")
    if data.shape[0] != len(case_ids):
        raise ValueError(f"values has {data.shape[0]} rows but case_ids has {len(case_ids)} entries")
    estimate = float(statistic(data))
    samples = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        sampled_cases = rng.integers(0, len(groups), size=len(groups))
        samples[replicate] = float(statistic(data[_resampled_indices(groups, sampled_cases)]))
    return _interval(
        estimate,
        samples,
        confidence_level=confidence_level,
        n_cases=len(labels),
        n_resamples=n_resamples,
        seed=seed,
    )


def paired_case_bootstrap(
    case_ids: Sequence[Hashable],
    left_values: NumericInput,
    right_values: NumericInput,
    *,
    statistic: Callable[[np.ndarray], float] = lambda sample: float(np.mean(sample)),
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = 2028,
) -> PairedBootstrapResult:
    """Return paired case-bootstrap intervals for left, right, and left-minus-right."""

    labels, groups, rng = _bootstrap_setup(
        case_ids, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed
    )
    left = _coerce_bootstrap_rows(left_values, name="left_values")
    right = _coerce_bootstrap_rows(right_values, name="right_values")
    if left.shape[0] != len(case_ids) or right.shape[0] != len(case_ids):
        raise ValueError("left_values and right_values must have one row per case_ids entry")
    left_estimate = float(statistic(left))
    right_estimate = float(statistic(right))
    left_samples = np.empty(n_resamples, dtype=np.float64)
    right_samples = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        sampled_cases = rng.integers(0, len(groups), size=len(groups))
        indices = _resampled_indices(groups, sampled_cases)
        left_samples[replicate] = float(statistic(left[indices]))
        right_samples[replicate] = float(statistic(right[indices]))
    difference_samples = left_samples - right_samples
    left_interval = _interval(left_estimate, left_samples, confidence_level=confidence_level,
                              n_cases=len(labels), n_resamples=n_resamples, seed=seed)
    right_interval = _interval(right_estimate, right_samples, confidence_level=confidence_level,
                               n_cases=len(labels), n_resamples=n_resamples, seed=seed)
    difference_interval = _interval(left_estimate - right_estimate, difference_samples,
                                    confidence_level=confidence_level, n_cases=len(labels),
                                    n_resamples=n_resamples, seed=seed)
    return PairedBootstrapResult(left_interval, right_interval, difference_interval)


def case_level_metric_bootstrap(
    case_ids: Sequence[Hashable],
    predictions: NumericInput,
    targets: NumericInput,
    metric: Metric,
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = 2028,
    atol: float = _DEFAULT_ATOL,
) -> BootstrapInterval:
    """Bootstrap any prediction/target metric while keeping cases intact."""

    labels, groups, rng = _bootstrap_setup(
        case_ids, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed
    )
    pred, target = _validated_pair(predictions, targets, atol=atol)
    if pred.shape[0] != len(case_ids):
        raise ValueError("predictions and targets must have one row per case_ids entry")
    estimate = float(metric(pred, target))
    samples = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        sampled_cases = rng.integers(0, len(groups), size=len(groups))
        indices = _resampled_indices(groups, sampled_cases)
        samples[replicate] = float(metric(pred[indices], target[indices]))
    return _interval(estimate, samples, confidence_level=confidence_level,
                     n_cases=len(labels), n_resamples=n_resamples, seed=seed)


def paired_case_metric_bootstrap(
    case_ids: Sequence[Hashable],
    left_predictions: NumericInput,
    right_predictions: NumericInput,
    targets: NumericInput,
    metric: Metric,
    *,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int | None = 2028,
    atol: float = _DEFAULT_ATOL,
) -> PairedBootstrapResult:
    """Bootstrap paired model metrics and their left-minus-right difference."""

    labels, groups, rng = _bootstrap_setup(
        case_ids, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed
    )
    left, left_target = _validated_pair(left_predictions, targets, atol=atol)
    right, right_target = _validated_pair(right_predictions, targets, atol=atol)
    if not np.array_equal(left_target, right_target):
        raise ValueError("left and right metrics must use identical targets")
    if left.shape[0] != len(case_ids) or right.shape[0] != len(case_ids):
        raise ValueError("predictions and targets must have one row per case_ids entry")
    left_estimate = float(metric(left, left_target))
    right_estimate = float(metric(right, right_target))
    left_samples = np.empty(n_resamples, dtype=np.float64)
    right_samples = np.empty(n_resamples, dtype=np.float64)
    for replicate in range(n_resamples):
        sampled_cases = rng.integers(0, len(groups), size=len(groups))
        indices = _resampled_indices(groups, sampled_cases)
        sampled_targets = left_target[indices]
        left_samples[replicate] = float(metric(left[indices], sampled_targets))
        right_samples[replicate] = float(metric(right[indices], sampled_targets))
    left_interval = _interval(left_estimate, left_samples, confidence_level=confidence_level,
                              n_cases=len(labels), n_resamples=n_resamples, seed=seed)
    right_interval = _interval(right_estimate, right_samples, confidence_level=confidence_level,
                               n_cases=len(labels), n_resamples=n_resamples, seed=seed)
    difference_interval = _interval(left_estimate - right_estimate, left_samples - right_samples,
                                    confidence_level=confidence_level, n_cases=len(labels),
                                    n_resamples=n_resamples, seed=seed)
    return PairedBootstrapResult(left_interval, right_interval, difference_interval)


__all__ = [
    "BootstrapInterval",
    "CalibrationBin",
    "Metric",
    "NumericInput",
    "PairedBootstrapResult",
    "TemperatureFitResult",
    "apply_temperature",
    "calibration_bins",
    "case_level_bootstrap",
    "case_level_metric_bootstrap",
    "ece",
    "expected_calibration_error",
    "fit_positive_temperature",
    "fit_temperature",
    "multiclass_brier",
    "paired_case_bootstrap",
    "paired_case_metric_bootstrap",
    "probabilities_from_logits",
    "score_expected_value_mae",
    "soft_accuracy",
    "soft_nll",
    "top1_accuracy",
    "validate_probabilities",
    "validate_probability_distributions",
]
