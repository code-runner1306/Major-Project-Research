"""Dynamic option-scoring models built on the project's pretrained GDN LM."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from config import PRODUCTION_MODELS
from model import GatedDeltaNetLM


class OptionScoringHead(nn.Module):
    """Shared pointer-style scorer for an arbitrary number of candidate spans."""

    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        if d_model < 1 or projection_dim < 1:
            raise ValueError("d_model and projection_dim must be positive")
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.query_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.option_proj = nn.Linear(d_model, projection_dim, bias=False)

    def forward(self, decision_state: torch.Tensor, option_states: torch.Tensor) -> torch.Tensor:
        """Return one logit per option.

        Args:
            decision_state: ``(d_model,)`` hidden state at the decision marker.
            option_states: ``(n_options, d_model)`` mean-pooled option states.
        """
        if decision_state.ndim != 1 or decision_state.numel() != self.d_model:
            raise ValueError("decision_state must be a d_model vector")
        if option_states.ndim != 2 or option_states.size(1) != self.d_model:
            raise ValueError("option_states must have shape (n_options, d_model)")
        if option_states.size(0) < 1:
            raise ValueError("At least one option is required")
        query = self.query_proj(decision_state)
        options = self.option_proj(option_states)
        return options @ query / (self.projection_dim ** 0.5)


class GDNDecisionModel(nn.Module):
    """A GatedDeltaNetLM backbone with a variable-option decision head.

    Examples are encoded one at a time. This intentionally avoids padding
    tokens changing the recurrent state or causal-attention representation.
    Call ``score`` per example and compute a list-wise soft-target loss.
    """

    def __init__(self, backbone: GatedDeltaNetLM, projection_dim: int = 128):
        super().__init__()
        if backbone.variant not in {"scalar", "channel-wise"}:
            raise ValueError(f"Unsupported decision backbone variant: {backbone.variant}")
        self.backbone = backbone
        # The LM head is tied to the token embedding and unnecessary for decisions.
        del self.backbone.output
        self.head = OptionScoringHead(backbone.d_model, projection_dim)
        self.projection_dim = projection_dim
        self.pretrained_checkpoint_sha256: str | None = None

    @property
    def variant(self) -> str:
        return self.backbone.variant

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode one unpadded token sequence to normalized hidden states."""
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.size(0) != 1 or token_ids.size(1) < 1:
            raise ValueError("encode expects one non-empty token sequence")
        hidden = self.backbone.tok_embeddings(token_ids)
        for block in self.backbone.blocks:
            hidden = block(hidden)
        return self.backbone.norm(hidden)[0]

    def score(
        self,
        token_ids: torch.Tensor,
        decision_index: int,
        option_spans: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """Score half-open token spans ``[(start, end), ...]`` in one sequence."""
        hidden = self.encode(token_ids)
        length = hidden.size(0)
        if not 0 <= decision_index < length:
            raise ValueError("decision_index is outside the token sequence")
        if not option_spans:
            raise ValueError("At least one option span is required")
        pooled = []
        previous_end = -1
        for start, end in option_spans:
            if not (0 <= start < end <= length):
                raise ValueError(f"Invalid option span {(start, end)} for sequence length {length}")
            if start < previous_end:
                raise ValueError("Option spans must not overlap and must be ordered")
            previous_end = end
            pooled.append(hidden[start:end].mean(dim=0))
        return self.head(hidden[decision_index], torch.stack(pooled))

    def score_batch(self, examples: Sequence[dict[str, object]]) -> list[torch.Tensor]:
        """Score unpadded examples and return logits with each example's own size."""
        return [
            self.score(
                example["input_ids"],  # type: ignore[arg-type]
                int(example["decision_index"]),
                example["option_spans"],  # type: ignore[arg-type]
            )
            for example in examples
        ]

    @classmethod
    def from_pretrained_checkpoint(
        cls,
        checkpoint_path: str | Path,
        variant: str,
        *,
        scale: str = "30M",
        projection_dim: int = 128,
        map_location: str | torch.device = "cpu",
    ) -> "GDNDecisionModel":
        if variant not in {"scalar", "channel-wise"}:
            raise ValueError("variant must be scalar or channel-wise")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        backbone = GatedDeltaNetLM(PRODUCTION_MODELS[scale], variant=variant, vocab_size=50_257)
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model_state"), dict):
            raise ValueError("Checkpoint must contain a model_state mapping")
        backbone.load_state_dict(payload["model_state"], strict=True)
        del payload
        model = cls(backbone, projection_dim=projection_dim)
        model.pretrained_checkpoint_sha256 = digest.hexdigest()
        return model

    def model_config(self) -> dict[str, object]:
        config = self.backbone.model_config
        return {
            "name": config.name,
            "d_model": config.d_model,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "d_ff": list(config.ffn_widths),
            "attention_layout": config.attention_layout,
            "variant": self.variant,
            "vocab_size": self.backbone.vocab_size,
            "projection_dim": self.projection_dim,
        }

    def save_inference_checkpoint(
        self,
        path: str | Path,
        *,
        run_fingerprint: str,
        selected_step: int,
        validation_nll: float,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Write model-only weights and return their SHA-256 checksum."""
        import os
        import tempfile

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "kind": "gdn_decision_model",
            "config": self.model_config(),
            "run_fingerprint": run_fingerprint,
            "selected_step": int(selected_step),
            "validation_nll": float(validation_nll),
            "metadata": {
                "pretrained_checkpoint_sha256": self.pretrained_checkpoint_sha256,
                **(metadata or {}),
            },
            "model_state": self.state_dict(),
        }
        handle, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            os.replace(temporary, destination)
            return digest
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load_inference_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        expected_fingerprint: str | None = None,
    ) -> "GDNDecisionModel":
        """Load a model-only checkpoint and optionally validate its run identity."""
        payload = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or payload.get("kind") != "gdn_decision_model":
            raise ValueError("Not a GDN decision-model checkpoint")
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported decision checkpoint schema")
        if expected_fingerprint is not None and payload.get("run_fingerprint") != expected_fingerprint:
            raise ValueError("Decision checkpoint run fingerprint does not match")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("Decision checkpoint is missing its model config")
        scale = str(config.get("name", "30M"))
        variant = str(config.get("variant", ""))
        if scale not in PRODUCTION_MODELS or variant not in {"scalar", "channel-wise"}:
            raise ValueError("Unsupported scale or variant in decision checkpoint")
        backbone = GatedDeltaNetLM(
            PRODUCTION_MODELS[scale], variant=variant, vocab_size=int(config.get("vocab_size", 50_257))
        )
        model = cls(backbone, projection_dim=int(config.get("projection_dim", 128)))
        state = payload.get("model_state")
        if not isinstance(state, dict):
            raise ValueError("Decision checkpoint is missing model_state")
        model.load_state_dict(state, strict=True)
        return model
