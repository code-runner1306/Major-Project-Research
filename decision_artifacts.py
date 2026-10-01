"""Crash-safe artifacts for the typed-decision fine-tuning pipeline.

The module deliberately has no eager PyTorch import.  JSON manifests and
lifecycle files can therefore be used in lightweight environments, while
checkpoint operations import PyTorch only when they are called.

The resume store writes a checkpoint to a local temporary directory first,
copies it to the durable output root, validates the copied file, and updates
the pointer last.  ``resume_A.pt`` and ``resume_B.pt`` are the only rolling
checkpoint slots retained by the store.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Literal
import uuid


SCHEMA_VERSION = 1
RESUME_SLOTS = ("A", "B")
LifecycleStatus = Literal[
    "pending",
    "training",
    "trained",
    "calibrating",
    "calibrated",
    "evaluating",
    "complete",
    "failed",
]

VALID_LIFECYCLE_STATUSES = frozenset(
    {
        "pending",
        "training",
        "trained",
        "calibrating",
        "calibrated",
        "evaluating",
        "complete",
        "failed",
    }
)

LIFECYCLE_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"pending", "training", "failed"}),
    "training": frozenset({"training", "trained", "failed"}),
    "trained": frozenset({"trained", "calibrating", "evaluating", "failed"}),
    "calibrating": frozenset({"calibrating", "calibrated", "failed"}),
    "calibrated": frozenset({"calibrated", "evaluating", "failed"}),
    "evaluating": frozenset({"evaluating", "complete", "failed"}),
    "complete": frozenset({"complete"}),
    # A failed stage may be explicitly retried by the orchestrator after it
    # has verified that the same run fingerprint is still in use.
    "failed": frozenset(
        {"failed", "training", "calibrating", "evaluating"}
    ),
}


class ArtifactError(RuntimeError):
    """Base class for artifact integrity and compatibility failures."""


class ResumeNotFoundError(ArtifactError):
    """Raised when no compatible valid rolling resume checkpoint exists."""


def _torch() -> Any:
    """Import PyTorch only for APIs that actually need it."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "PyTorch is required for checkpoint operations; install a compatible "
            "PyTorch build before loading or saving model artifacts."
        ) from exc
    return torch


def utc_now() -> str:
    """Return a UTC timestamp suitable for lifecycle and artifact metadata."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonicalize(value: Any) -> Any:
    """Convert supported configuration values into deterministic JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return {"__type__": "bytes", "hex": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite floats cannot be included in canonical JSON")
        # JSON has one representation for negative zero in some encoders and
        # another in others.  Treat both as the same configuration value.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Canonical configuration mappings must use string keys; "
                    f"got {type(key).__name__}"
                )
            result[key] = _canonicalize(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=lambda item: canonical_json(item))

    # Support simple scalar wrappers such as NumPy scalar values without
    # making NumPy a dependency of the artifact module.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _canonicalize(item_method())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"Cannot canonicalize {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a value deterministically for hashing and identity checks."""

    return json.dumps(
        _canonicalize(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def config_fingerprint(
    config: Any,
    *,
    schema_version: int = SCHEMA_VERSION,
) -> str:
    """Return the canonical SHA-256 identity of a research configuration."""

    envelope = {"schema_version": schema_version, "config": config}
    return sha256(canonical_json(envelope).encode("utf-8")).hexdigest()


# Descriptive aliases make call sites read naturally without creating a second
# fingerprint implementation.
canonical_fingerprint = config_fingerprint
fingerprint = config_fingerprint


def _json_ready(value: Any) -> Any:
    """Prepare JSON data while preserving deterministic handling of paths."""

    return _canonicalize(value)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability; Windows does not expose this cleanly."""

    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except (OSError, ValueError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    """Atomically replace a JSON object, preserving the previous valid file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        _json_ready(data),
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def read_json_object(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a JSON object and reject arrays/scalars at the artifact boundary."""

    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ArtifactError(f"Expected a JSON object in {target}")
    return value


def checkpoint_sha256(path: str | os.PathLike[str]) -> str:
    """Hash a checkpoint or other artifact in bounded memory."""

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


sha256_file = checkpoint_sha256


def lifecycle_path(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / "lifecycle.json"


def initialize_lifecycle(
    run_dir: str | os.PathLike[str],
    run_fingerprint: str,
    *,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a pending lifecycle document atomically."""

    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("run_fingerprint must be a non-empty string")
    timestamp = utc_now()
    lifecycle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "fingerprint": run_fingerprint,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    if details is not None:
        lifecycle["details"] = dict(details)
    atomic_write_json(lifecycle_path(run_dir), lifecycle)
    return lifecycle


def transition_lifecycle(
    run_dir: str | os.PathLike[str],
    status: LifecycleStatus,
    run_fingerprint: str,
    *,
    details: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically record a validated lifecycle transition.

    ``failure`` is stored separately for a failed stage and is intended to
    contain fields such as ``exception_type``, ``reason``, ``stage``, and
    ``step``.  The existing lifecycle file is never replaced if validation or
    serialization fails.
    """

    if status not in VALID_LIFECYCLE_STATUSES:
        raise ValueError(f"Unknown lifecycle status: {status!r}")
    path = lifecycle_path(run_dir)
    if path.exists():
        current = read_json_object(path)
    else:
        current = initialize_lifecycle(run_dir, run_fingerprint)

    if current.get("fingerprint") != run_fingerprint:
        raise ArtifactError(
            "Lifecycle fingerprint differs; refusing to reuse an incompatible run directory"
        )
    previous = current.get("status", "pending")
    if previous not in VALID_LIFECYCLE_STATUSES:
        raise ArtifactError(f"Lifecycle contains unknown status: {previous!r}")
    if status not in LIFECYCLE_TRANSITIONS[previous]:
        raise ArtifactError(f"Invalid lifecycle transition {previous!r} -> {status!r}")
    if failure is not None and status != "failed":
        raise ValueError("failure details are only valid for the failed status")

    lifecycle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "fingerprint": run_fingerprint,
        "created_at": current.get("created_at", utc_now()),
        "updated_at": utc_now(),
    }
    if details is not None:
        lifecycle["details"] = dict(details)
    elif isinstance(current.get("details"), Mapping):
        lifecycle["details"] = dict(current["details"])
    if failure is not None:
        lifecycle["failure"] = dict(failure)
    elif status != "failed":
        lifecycle.pop("failure", None)
    atomic_write_json(path, lifecycle)
    return lifecycle


def read_lifecycle(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    return read_json_object(lifecycle_path(run_dir))


def _to_cpu(value: Any) -> Any:
    """Move tensor-like values to CPU without importing Torch eagerly."""

    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _module_state_dict(module: Any | None) -> dict[str, Any] | None:
    if module is None:
        return None
    state = module.state_dict()
    if not isinstance(state, Mapping):
        raise ArtifactError("Module state_dict() must return a mapping")
    return _to_cpu(state)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, optional NumPy, CPU Torch, and CUDA Torch RNG states."""

    torch = _torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    torch = _torch()
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def build_resume_payload(
    *,
    model: Any,
    head: Any | None = None,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int,
    completed_optimizer_step: int,
    gradient_accumulation_step: int = 0,
    sampler_state: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
    best_validation_state: Mapping[str, Any] | None = None,
    run_fingerprint: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete optimizer-boundary payload from live training objects."""

    if epoch < 0 or completed_optimizer_step < 0 or gradient_accumulation_step < 0:
        raise ValueError("epoch and step values must be non-negative")
    if model is None:
        raise ValueError("model is required")
    payload: dict[str, Any] = {
        "artifact_type": "decision_resume_checkpoint",
        "schema_version": SCHEMA_VERSION,
        "model_state": _module_state_dict(model),
        "head_state": _module_state_dict(head),
        "optimizer_state": _to_cpu(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler_state": _to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "scaler_state": _to_cpu(scaler.state_dict()) if scaler is not None else None,
        "epoch": epoch,
        "completed_optimizer_step": completed_optimizer_step,
        "gradient_accumulation_step": gradient_accumulation_step,
        "sampler_state": dict(sampler_state or {}),
        "rng_state": dict(rng_state or capture_rng_state()),
        "best_validation_state": dict(best_validation_state or {}),
        "saved_at": utc_now(),
    }
    if run_fingerprint is not None:
        payload["run_fingerprint"] = run_fingerprint
    if extra is not None:
        payload["extra"] = dict(extra)
    return payload


def _torch_save_atomic(payload: Any, target: Path) -> None:
    torch = _torch()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        torch.save(payload, temporary)
        _fsync_file(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _torch_load(path: Path, *, map_location: Any = "cpu", weights_only: bool) -> Any:
    torch = _torch()
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # Older supported Torch releases do not expose weights_only.  The
        # fallback is only needed for compatibility with those releases.
        return torch.load(path, map_location=map_location)


def _copy_file_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _slot_name(slot: str) -> str:
    if slot not in RESUME_SLOTS:
        raise ValueError(f"Unknown resume slot {slot!r}; expected A or B")
    return slot


@dataclass(frozen=True)
class ResumeLoadResult:
    """Validated rolling checkpoint selected for resumption."""

    payload: Mapping[str, Any]
    slot: str
    path: Path
    sha256: str
    completed_optimizer_step: int
    fallback_used: bool
    rejected_slots: Mapping[str, str]


class ResumeCheckpointStore:
    """Alternating, checksummed, externally-stored resume checkpoints."""

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        *,
        local_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.local_root = (
            Path(local_root)
            if local_root is not None
            else Path(tempfile.gettempdir()) / "decision-artifacts"
        )

    @property
    def pointer_path(self) -> Path:
        return self.output_root / "latest_resume.json"

    def slot_path(self, slot: str) -> Path:
        return self.output_root / f"resume_{_slot_name(slot)}.pt"

    def slot_metadata_path(self, slot: str) -> Path:
        return self.output_root / f"resume_{_slot_name(slot)}.meta.json"

    def _pointer_slot(self) -> str | None:
        try:
            pointer = read_json_object(self.pointer_path)
            slot = pointer.get("slot")
            return slot if slot in RESUME_SLOTS else None
        except (OSError, ValueError, json.JSONDecodeError, ArtifactError):
            return None

    def _validate_slot(
        self,
        slot: str,
        *,
        expected_fingerprint: str | None = None,
        expected_step: int | None = None,
        expected_sha256: str | None = None,
        map_location: Any = "cpu",
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
        slot = _slot_name(slot)
        path = self.slot_path(slot)
        metadata_path = self.slot_metadata_path(slot)
        try:
            metadata = read_json_object(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError, ArtifactError) as exc:
            return None, None, f"metadata is missing or invalid: {exc}"
        if metadata.get("slot") != slot or metadata.get("filename") != path.name:
            return None, metadata, "slot metadata names a different file"
        if expected_fingerprint is not None and metadata.get("fingerprint") != expected_fingerprint:
            return None, metadata, "run fingerprint differs"
        if expected_sha256 is not None and metadata.get("sha256") != expected_sha256:
            return None, metadata, "pointer checksum differs from slot metadata"
        if not path.is_file():
            return None, metadata, "checkpoint file is missing"
        actual_sha256 = checkpoint_sha256(path)
        if metadata.get("sha256") != actual_sha256:
            return None, metadata, "checkpoint checksum is invalid"
        try:
            payload = _torch_load(path, map_location=map_location, weights_only=False)
        except Exception as exc:  # Torch has several loader-specific errors.
            return None, metadata, f"checkpoint cannot be loaded: {exc}"
        if not isinstance(payload, dict):
            return None, metadata, "checkpoint payload is not a dictionary"
        if payload.get("artifact_type") != "decision_resume_checkpoint":
            return None, metadata, "checkpoint artifact type is invalid"
        if expected_fingerprint is not None and payload.get("run_fingerprint") != expected_fingerprint:
            return None, metadata, "checkpoint payload fingerprint differs"
        step = payload.get("completed_optimizer_step")
        if not isinstance(step, int) or step < 0:
            return None, metadata, "completed optimizer step is missing or invalid"
        if expected_step is not None and step != expected_step:
            return None, metadata, "completed optimizer step differs"
        return payload, metadata, None

    def save(
        self,
        payload: Mapping[str, Any],
        *,
        run_fingerprint: str | None = None,
        completed_optimizer_step: int | None = None,
    ) -> dict[str, Any]:
        """Save a resume payload to the inactive slot and advance the pointer."""

        if not isinstance(payload, Mapping):
            raise TypeError("resume payload must be a mapping")
        candidate = dict(payload)
        candidate.setdefault("artifact_type", "decision_resume_checkpoint")
        candidate.setdefault("schema_version", SCHEMA_VERSION)
        if candidate["artifact_type"] != "decision_resume_checkpoint":
            raise ArtifactError("resume payload has an incompatible artifact_type")
        if run_fingerprint is not None:
            existing = candidate.get("run_fingerprint")
            if existing is not None and existing != run_fingerprint:
                raise ArtifactError("resume payload fingerprint differs from requested fingerprint")
            candidate["run_fingerprint"] = run_fingerprint
        step = candidate.get("completed_optimizer_step")
        if completed_optimizer_step is not None:
            if step is not None and step != completed_optimizer_step:
                raise ArtifactError("resume payload step differs from requested step")
            step = completed_optimizer_step
            candidate["completed_optimizer_step"] = step
        if not isinstance(step, int) or step < 0:
            raise ValueError("resume payload needs a non-negative completed_optimizer_step")
        if run_fingerprint is not None and (
            not isinstance(candidate.get("run_fingerprint"), str)
            or not candidate["run_fingerprint"]
        ):
            raise ValueError("run_fingerprint must be a non-empty string")

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.local_root.mkdir(parents=True, exist_ok=True)
        pointer_slot = self._pointer_slot()
        active: str | None = None
        if pointer_slot is not None:
            pointer_payload, _, _ = self._validate_slot(
                pointer_slot, expected_fingerprint=run_fingerprint
            )
            if pointer_payload is not None:
                active = pointer_slot
        if active is None:
            valid_slots: list[tuple[str, int]] = []
            for slot in RESUME_SLOTS:
                valid_payload, _, _ = self._validate_slot(
                    slot, expected_fingerprint=run_fingerprint
                )
                if valid_payload is not None:
                    valid_slots.append((slot, valid_payload["completed_optimizer_step"]))
            if valid_slots:
                active = max(valid_slots, key=lambda item: item[1])[0]
        target_slot = "A" if active is None else ("B" if active == "A" else "A")
        target = self.slot_path(target_slot)
        local_temp = self.local_root / (
            f".{target.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            _torch_save_atomic(candidate, local_temp)
            local_checksum = checkpoint_sha256(local_temp)
            _copy_file_atomic(local_temp, target)
            durable_checksum = checkpoint_sha256(target)
            if durable_checksum != local_checksum:
                raise ArtifactError("durable resume checkpoint checksum differs from local copy")
            loaded = _torch_load(target, map_location="cpu", weights_only=False)
            if not isinstance(loaded, dict) or loaded.get("artifact_type") != "decision_resume_checkpoint":
                raise ArtifactError("durable resume checkpoint failed payload validation")
            if loaded.get("completed_optimizer_step") != step:
                raise ArtifactError("durable resume checkpoint failed step validation")
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "resume_slot_metadata",
                "slot": target_slot,
                "filename": target.name,
                "sha256": durable_checksum,
                "bytes": target.stat().st_size,
                "fingerprint": candidate.get("run_fingerprint"),
                "completed_optimizer_step": step,
                "saved_at": candidate.get("saved_at", utc_now()),
            }
            atomic_write_json(self.slot_metadata_path(target_slot), metadata)
            pointer = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "latest_resume_pointer",
                "slot": target_slot,
                "filename": target.name,
                "sha256": durable_checksum,
                "fingerprint": candidate.get("run_fingerprint"),
                "completed_optimizer_step": step,
                "updated_at": utc_now(),
            }
            atomic_write_json(self.pointer_path, pointer)
            return {
                "slot": target_slot,
                "path": str(target),
                "metadata_path": str(self.slot_metadata_path(target_slot)),
                "pointer_path": str(self.pointer_path),
                "sha256": durable_checksum,
                "completed_optimizer_step": step,
            }
        finally:
            try:
                local_temp.unlink()
            except FileNotFoundError:
                pass

    def load_latest(
        self,
        *,
        expected_fingerprint: str | None = None,
        map_location: Any = "cpu",
        repair_pointer: bool = True,
    ) -> ResumeLoadResult:
        """Load the newest compatible slot, falling back to the other slot."""

        rejected: dict[str, str] = {}
        pointer_slot: str | None = None
        pointer_checksum: str | None = None
        try:
            pointer = read_json_object(self.pointer_path)
            if pointer.get("slot") in RESUME_SLOTS:
                pointer_slot = pointer["slot"]
                pointer_checksum = pointer.get("sha256")
                if not isinstance(pointer_checksum, str) or not pointer_checksum:
                    rejected["pointer"] = "pointer checksum is missing or invalid"
            else:
                rejected["pointer"] = "pointer has an invalid slot"
        except (OSError, ValueError, json.JSONDecodeError, ArtifactError) as exc:
            rejected["pointer"] = f"pointer is missing or invalid: {exc}"

        candidates: list[str] = []
        if pointer_slot is not None:
            candidates.append(pointer_slot)
        candidates.extend(slot for slot in RESUME_SLOTS if slot not in candidates)
        valid: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        pointer_valid = False
        for slot in candidates:
            payload, metadata, reason = self._validate_slot(
                slot,
                expected_fingerprint=expected_fingerprint,
                expected_sha256=(
                    pointer_checksum
                    if slot == pointer_slot and isinstance(pointer_checksum, str)
                    else None
                ),
                map_location=map_location,
            )
            if payload is not None and metadata is not None:
                valid[slot] = (payload, metadata)
                pointer_valid = (
                    slot == pointer_slot
                    and isinstance(pointer_checksum, str)
                    and bool(pointer_checksum)
                )
            elif reason is not None:
                rejected[slot] = reason

        # If the pointer itself was stale or corrupted, validate its named
        # slot using the durable slot metadata as a fallback.  This permits
        # recovery when the pointer write was the part that was interrupted.
        if pointer_slot is not None and not pointer_valid:
            payload, metadata, reason = self._validate_slot(
                pointer_slot,
                expected_fingerprint=expected_fingerprint,
                map_location=map_location,
            )
            if payload is not None and metadata is not None:
                valid[pointer_slot] = (payload, metadata)
            elif reason is not None:
                rejected[pointer_slot] = reason

        if not valid:
            detail = "; ".join(f"{key}: {value}" for key, value in rejected.items())
            raise ResumeNotFoundError(f"No valid compatible resume checkpoint found ({detail})")

        if pointer_valid and pointer_slot in valid:
            selected_slot = pointer_slot
        else:
            selected_slot = max(
                valid,
                key=lambda slot: (
                    int(valid[slot][0].get("completed_optimizer_step", -1)),
                    str(valid[slot][1].get("saved_at", "")),
                ),
            )
        payload, metadata = valid[selected_slot]
        fallback_used = selected_slot != pointer_slot
        if fallback_used and repair_pointer:
            repaired_pointer = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "latest_resume_pointer",
                "slot": selected_slot,
                "filename": self.slot_path(selected_slot).name,
                "sha256": metadata["sha256"],
                "fingerprint": metadata.get("fingerprint"),
                "completed_optimizer_step": payload["completed_optimizer_step"],
                "updated_at": utc_now(),
                "repaired_from_fallback": True,
            }
            atomic_write_json(self.pointer_path, repaired_pointer)
        return ResumeLoadResult(
            payload=payload,
            slot=selected_slot,
            path=self.slot_path(selected_slot),
            sha256=str(metadata["sha256"]),
            completed_optimizer_step=int(payload["completed_optimizer_step"]),
            fallback_used=fallback_used,
            rejected_slots=rejected,
        )


def save_resume_checkpoint(
    output_root: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    local_root: str | os.PathLike[str] | None = None,
    run_fingerprint: str | None = None,
    completed_optimizer_step: int | None = None,
) -> dict[str, Any]:
    """Functional wrapper around :class:`ResumeCheckpointStore.save`."""

    return ResumeCheckpointStore(output_root, local_root=local_root).save(
        payload,
        run_fingerprint=run_fingerprint,
        completed_optimizer_step=completed_optimizer_step,
    )


def load_latest_resume_checkpoint(
    output_root: str | os.PathLike[str],
    *,
    expected_fingerprint: str | None = None,
    map_location: Any = "cpu",
    local_root: str | os.PathLike[str] | None = None,
    repair_pointer: bool = True,
) -> ResumeLoadResult:
    """Functional wrapper around :class:`ResumeCheckpointStore.load_latest`."""

    return ResumeCheckpointStore(output_root, local_root=local_root).load_latest(
        expected_fingerprint=expected_fingerprint,
        map_location=map_location,
        repair_pointer=repair_pointer,
    )


def _model_only_payload(
    *,
    model: Any,
    head: Any | None,
    model_config: Mapping[str, Any],
    label_schema: Any,
    source_run_fingerprint: str,
    selected_step: int,
    validation_summary: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not source_run_fingerprint:
        raise ValueError("source_run_fingerprint must be non-empty")
    if selected_step < 0:
        raise ValueError("selected_step must be non-negative")
    return {
        "artifact_type": "decision_model_only_checkpoint",
        "schema_version": SCHEMA_VERSION,
        "model_config": _json_ready(model_config),
        "label_schema": _json_ready(label_schema),
        "source_run_fingerprint": source_run_fingerprint,
        "model_state": _module_state_dict(model),
        "head_state": _module_state_dict(head),
        "selected_step": selected_step,
        "validation_summary": _json_ready(validation_summary),
        "metadata": _json_ready(metadata or {}),
        "saved_at": utc_now(),
    }


def model_checksum_sidecar(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    return target.with_name(target.name + ".sha256.json")


def save_model_only_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: Any,
    head: Any | None = None,
    model_config: Mapping[str, Any],
    label_schema: Any,
    source_run_fingerprint: str,
    selected_step: int,
    validation_summary: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a checksummed inference-only model/head artifact."""

    target = Path(path)
    payload = _model_only_payload(
        model=model,
        head=head,
        model_config=model_config,
        label_schema=label_schema,
        source_run_fingerprint=source_run_fingerprint,
        selected_step=selected_step,
        validation_summary=validation_summary,
        metadata=metadata,
    )
    _torch_save_atomic(payload, target)
    digest = checkpoint_sha256(target)
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "model_only_checksum",
        "filename": target.name,
        "sha256": digest,
        "bytes": target.stat().st_size,
        "source_run_fingerprint": source_run_fingerprint,
        "selected_step": selected_step,
        "created_at": utc_now(),
    }
    sidecar_path = model_checksum_sidecar(target)
    atomic_write_json(sidecar_path, sidecar)
    return {
        "path": str(target),
        "checksum_path": str(sidecar_path),
        "sha256": digest,
        "bytes": target.stat().st_size,
        "selected_step": selected_step,
    }


def save_best_model_checkpoint(path: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
    """Named convenience wrapper for the selected validation-best export."""

    return save_model_only_checkpoint(path, **kwargs)


def save_final_model_checkpoint(path: str | os.PathLike[str], **kwargs: Any) -> dict[str, Any]:
    """Named convenience wrapper for the completed-budget export."""

    return save_model_only_checkpoint(path, **kwargs)


def load_model_only_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_source_run_fingerprint: str | None = None,
    expected_model_config: Mapping[str, Any] | None = None,
    map_location: Any = "cpu",
    require_checksum: bool = True,
) -> dict[str, Any]:
    """Validate and load an inference-only checkpoint without optimizer state."""

    target = Path(path)
    if not target.is_file():
        raise ArtifactError(f"Model-only checkpoint is missing: {target}")
    sidecar_path = model_checksum_sidecar(target)
    if require_checksum:
        try:
            sidecar = read_json_object(sidecar_path)
        except (OSError, ValueError, json.JSONDecodeError, ArtifactError) as exc:
            raise ArtifactError(f"Model-only checksum sidecar is invalid: {exc}") from exc
        if sidecar.get("filename") != target.name:
            raise ArtifactError("Model-only checksum sidecar names a different file")
        digest = checkpoint_sha256(target)
        if sidecar.get("sha256") != digest:
            raise ArtifactError("Model-only checkpoint checksum is invalid")

    try:
        payload = _torch_load(target, map_location=map_location, weights_only=True)
    except Exception as exc:  # Torch loader errors vary by version.
        raise ArtifactError(f"Model-only checkpoint cannot be loaded: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("Model-only checkpoint payload is not a dictionary")
    if payload.get("artifact_type") != "decision_model_only_checkpoint":
        raise ArtifactError("Model-only checkpoint artifact type is invalid")
    source_fingerprint = payload.get("source_run_fingerprint")
    if expected_source_run_fingerprint is not None and source_fingerprint != expected_source_run_fingerprint:
        raise ArtifactError("Model-only checkpoint source fingerprint differs")
    if expected_model_config is not None:
        actual_config = payload.get("model_config")
        if canonical_json(actual_config) != canonical_json(expected_model_config):
            raise ArtifactError("Model-only checkpoint model configuration differs")
    if not isinstance(payload.get("model_state"), Mapping):
        raise ArtifactError("Model-only checkpoint lacks model_state")
    if payload.get("head_state") is not None and not isinstance(payload.get("head_state"), Mapping):
        raise ArtifactError("Model-only checkpoint head_state is invalid")
    return payload


__all__ = [
    "ArtifactError",
    "LifecycleStatus",
    "LIFECYCLE_TRANSITIONS",
    "RESUME_SLOTS",
    "ResumeCheckpointStore",
    "ResumeLoadResult",
    "ResumeNotFoundError",
    "SCHEMA_VERSION",
    "atomic_write_json",
    "build_resume_payload",
    "canonical_fingerprint",
    "canonical_json",
    "capture_rng_state",
    "checkpoint_sha256",
    "config_fingerprint",
    "fingerprint",
    "initialize_lifecycle",
    "lifecycle_path",
    "load_latest_resume_checkpoint",
    "load_model_only_checkpoint",
    "model_checksum_sidecar",
    "read_json_object",
    "read_lifecycle",
    "restore_rng_state",
    "save_best_model_checkpoint",
    "save_final_model_checkpoint",
    "save_model_only_checkpoint",
    "save_resume_checkpoint",
    "sha256_file",
    "transition_lifecycle",
    "utc_now",
]
