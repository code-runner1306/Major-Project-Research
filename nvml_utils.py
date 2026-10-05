"""Compatibility helpers for NVML values returned by pynvml."""

from __future__ import annotations

from typing import Any


def nvml_text(value: Any, encoding: str = "utf-8") -> str:
    """Normalize NVML text, which pynvml versions may return as bytes or str."""

    if isinstance(value, bytes):
        return value.decode(encoding, errors="replace")
    return str(value)
