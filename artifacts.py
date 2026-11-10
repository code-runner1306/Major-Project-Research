"""Schema-versioned manifests, atomic lifecycle state, and strict resumption."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Literal, Mapping

import torch


SCHEMA_VERSION = 1
LifecycleStatus = Literal["pending", "training", "trained", "evaluating", "complete", "failed"]
VALID_TRANSITIONS = {
    "pending": {"training", "failed"},
    "training": {"trained", "failed"},
    "trained": {"evaluating", "failed"},
    "evaluating": {"complete", "failed"},
    "complete": set(),
    # A failed run may retry evaluation only after orchestration validates that
    # its completed-training metrics and checkpoint are intact.
    "failed": {"evaluating"},
}
MANDATORY_METRICS = {
    "tinystories": ("final_validation", "benchmarks", "energy", "gate_diagnostics", "static_cost"),
    "mqar": ("retrieval", "energy", "static_cost"),
    "overwrite": ("retrieval", "energy", "static_cost"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(data: Any) -> str:
    return json.dumps(data, default=_json_default, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def fingerprint(data: Mapping[str, Any]) -> str:
    return sha256(canonical_json(data).encode("utf-8")).hexdigest()


def atomic_write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, default=_json_default, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def source_revision_state(workspace: str | Path = ".") -> dict[str, str | bool]:
    """Record enough revision information to reject artifacts from changed code."""

    root = Path(workspace)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, text=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"revision": "unavailable", "dirty": True}


def runtime_environment() -> dict[str, Any]:
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


@dataclass(frozen=True)
class SeedManifest:
    family: str
    run_mode: str
    scale: str
    variant: str
    model: Mapping[str, Any]
    seeds: Mapping[str, int]
    budget: Mapping[str, Any]
    data_identity: Mapping[str, Any]
    evaluation_protocol: Mapping[str, Any]
    expected_artifacts: tuple[str, ...]
    environment: Mapping[str, Any] = field(default_factory=runtime_environment)
    source: Mapping[str, Any] = field(default_factory=source_revision_state)
    schema_version: int = SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_artifacts"] = list(self.expected_artifacts)
        return data

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload()
        payload["fingerprint"] = fingerprint(payload)
        return payload


def write_manifest(path: str | Path, manifest: SeedManifest) -> dict[str, Any]:
    data = manifest.to_dict()
    atomic_write_json(path, data)
    return data


def _recurrence_drift(actual: Mapping[str, Any], expected: SeedManifest) -> str | None:
    """Fail-closed backend/build/dtype/chunk/qualification comparison.

    The canonical fingerprint comparison below already rejects any drift, but
    this explicit guard keeps optimized-production resumption fail-closed with
    a stable message when recurrence provenance is missing or differs.
    """
    expected_model = expected.model
    actual_recurrence = actual.get("model", {}).get("recurrence") if isinstance(actual.get("model"), Mapping) else None
    expected_recurrence = expected_model.get("recurrence") if isinstance(expected_model, Mapping) else None
    if expected_recurrence is None:
        return None
    if not isinstance(actual_recurrence, Mapping):
        return "configuration/environment/source fingerprint differs"
    keys = (
        "requested_backend", "semantic_version", "implementation_source_hash",
        "input_dtype", "state_accumulator_dtype", "chunk_size",
        "qualification_artifact", "qualification_hash",
        "compiled_architectures", "chunk_policy",
    )
    for key in keys:
        if actual_recurrence.get(key) != expected_recurrence.get(key):
            return "configuration/environment/source fingerprint differs"
    return None


def manifest_compatibility(
    path: str | Path, expected: SeedManifest, *, allow_source_drift: bool = False,
) -> tuple[bool, str]:
    try:
        actual = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}"
    if actual.get("schema_version") != SCHEMA_VERSION:
        return False, "schema version is missing or incompatible (legacy artifact)"
    stored_fingerprint = actual.get("fingerprint")
    actual_payload = {key: value for key, value in actual.items() if key != "fingerprint"}
    if not isinstance(stored_fingerprint, str) or stored_fingerprint != fingerprint(actual_payload):
        return False, "manifest fingerprint does not match manifest contents"
    if actual.get("run_mode") != expected.run_mode:
        return False, f"run mode differs ({actual.get('run_mode')} vs {expected.run_mode})"
    drift = _recurrence_drift(actual, expected)
    if drift is not None:
        return False, drift
    expected_payload = expected.payload()
    actual_comparable = dict(actual_payload)
    expected_comparable = dict(expected_payload)
    if allow_source_drift:
        actual_comparable.pop("source", None)
        expected_comparable.pop("source", None)
    if actual_comparable != expected_comparable:
        return False, "configuration/environment/source fingerprint differs"
    return True, "compatible"


def lifecycle_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "lifecycle.json"


def initialize_lifecycle(run_dir: str | Path, manifest_fingerprint: str) -> dict[str, Any]:
    data = {"schema_version": SCHEMA_VERSION, "status": "pending", "fingerprint": manifest_fingerprint, "updated_at": utc_now()}
    atomic_write_json(lifecycle_path(run_dir), data)
    return data


def transition_lifecycle(
    run_dir: str | Path,
    status: LifecycleStatus,
    manifest_fingerprint: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = lifecycle_path(run_dir)
    current = read_json(path) if path.exists() else initialize_lifecycle(run_dir, manifest_fingerprint)
    if current.get("fingerprint") != manifest_fingerprint:
        raise ValueError("Lifecycle fingerprint differs; refusing to reuse incompatible run directory")
    previous = current.get("status", "pending")
    if status != previous and status not in VALID_TRANSITIONS.get(previous, set()):
        raise ValueError(f"Invalid lifecycle transition {previous!r} -> {status!r}")
    data = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "fingerprint": manifest_fingerprint,
        "updated_at": utc_now(),
    }
    if details:
        data["details"] = dict(details)
    atomic_write_json(path, data)
    return data


def checkpoint_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    completed_steps: int,
    processed_tokens: int,
) -> dict[str, Any]:
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "completed_steps": completed_steps,
        "processed_tokens": processed_tokens,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, target)
    return {
        "path": target.name,
        "sha256": checkpoint_sha256(target),
        "completed_steps": completed_steps,
        "processed_tokens": processed_tokens,
    }


def validate_checkpoint(path: str | Path, expected_steps: int, expected_tokens: int) -> tuple[bool, str, dict[str, Any] | None]:
    target = Path(path)
    if not target.is_file():
        return False, "checkpoint is missing", None
    try:
        checkpoint = torch.load(target, map_location="cpu", weights_only=False)
    except Exception as exc:  # torch emits several loader exception types
        return False, f"checkpoint cannot be loaded: {exc}", None
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        return False, "checkpoint lacks model state", None
    if checkpoint.get("completed_steps") != expected_steps:
        return False, "checkpoint completed-step count differs", checkpoint
    if checkpoint.get("processed_tokens") != expected_tokens:
        return False, "checkpoint processed-token total differs", checkpoint
    return True, "valid", checkpoint


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def validate_completion(
    run_dir: str | Path, expected: SeedManifest, *, allow_source_drift: bool = False,
) -> tuple[bool, str]:
    """Validate every required primary artifact before a run may be resumed/skipped."""

    root = Path(run_dir)
    compatible, reason = manifest_compatibility(
        root / "manifest.json", expected, allow_source_drift=allow_source_drift,
    )
    if not compatible:
        return False, reason
    try:
        stored_manifest = read_json(root / "manifest.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid manifest: {exc}"
    manifest = expected.to_dict()
    try:
        lifecycle = read_json(root / "lifecycle.json")
        metrics = read_json(root / "metrics.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"missing or corrupt lifecycle/metrics: {exc}"
    if lifecycle.get("status") != "complete":
        return False, f"lifecycle is {lifecycle.get('status')!r}, not complete"
    expected_lifecycle_fingerprint = (
        stored_manifest.get("fingerprint") if allow_source_drift else manifest["fingerprint"]
    )
    if lifecycle.get("fingerprint") != expected_lifecycle_fingerprint:
        return False, "lifecycle fingerprint differs"
    required = MANDATORY_METRICS.get(expected.family, ())
    missing = [key for key in required if key not in metrics]
    if missing:
        return False, f"mandatory metrics missing: {', '.join(missing)}"
    if _contains_nonfinite(metrics):
        return False, "metrics contain non-finite values"
    checkpoint_info = metrics.get("final_checkpoint")
    if not isinstance(checkpoint_info, Mapping):
        return False, "final checkpoint metadata missing"
    budget = expected.budget
    steps = budget.get("max_optimizer_steps")
    expected_tokens = metrics.get("training", {}).get("processed_tokens") if isinstance(metrics.get("training"), Mapping) else None
    if not isinstance(steps, int) or not isinstance(expected_tokens, int):
        return False, "fixed training budget metadata missing"
    valid, checkpoint_reason, _ = validate_checkpoint(root / str(checkpoint_info.get("path", "")), steps, expected_tokens)
    if not valid:
        return False, checkpoint_reason
    if checkpoint_info.get("sha256") != checkpoint_sha256(root / str(checkpoint_info["path"])):
        return False, "checkpoint checksum differs"
    return True, "complete"
