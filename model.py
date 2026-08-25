"""Gated DeltaNet language model with cache-aware inference APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


LayerCache = dict[str, torch.Tensor | str]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalConv1d(nn.Module):
    """Depthwise causal convolution with a cache of projected input values."""

    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=True)

    @property
    def cache_length(self) -> int:
        return self.kernel_size - 1

    def _validate_cache(self, cache: torch.Tensor, x: torch.Tensor) -> None:
        expected = (x.size(0), self.cache_length, self.channels)
        if tuple(cache.shape) != expected:
            raise ValueError(f"Invalid convolution cache shape {tuple(cache.shape)}; expected {expected}")
        if cache.device != x.device or cache.dtype != x.dtype:
            raise ValueError("Convolution cache must match input device and dtype")

    def _cache_from_inputs(self, x: torch.Tensor) -> torch.Tensor:
        recent = x[:, -self.cache_length :]
        if recent.size(1) == self.cache_length:
            return recent
        return F.pad(recent, (0, 0, self.cache_length - recent.size(1), 0))

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        *,
        return_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("CausalConv1d expects (batch, time, channels)")
        if x.size(-1) != self.channels:
            raise ValueError("Input channel count does not match convolution")
        if self.cache_length == 0:
            out = self.conv(x.transpose(1, 2)).transpose(1, 2)
            new_cache = x[:, :0]
        elif cache is None:
            out = self.conv(F.pad(x.transpose(1, 2), (self.cache_length, 0))).transpose(1, 2)
            new_cache = self._cache_from_inputs(x)
        else:
            self._validate_cache(cache, x)
            history = torch.cat((cache, x), dim=1)
            out = self.conv(history.transpose(1, 2)).transpose(1, 2)
            new_cache = history[:, -self.cache_length :]
        return (out, new_cache) if return_cache else out


def recurrent_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential DeltaNet recurrence, optionally continuing a cached state."""

    batch, time, heads, width = q.shape
    expected_state = (batch, heads, width, width)
    if initial_state is None:
        state = torch.zeros(expected_state, device=q.device, dtype=q.dtype)
    else:
        if tuple(initial_state.shape) != expected_state:
            raise ValueError(f"Invalid recurrent state shape {tuple(initial_state.shape)}; expected {expected_state}")
        if initial_state.device != q.device or initial_state.dtype != q.dtype:
            raise ValueError("Recurrent state must match Q device and dtype")
        state = initial_state

    q = F.normalize(q, p=2, dim=-1)
    k = F.normalize(k, p=2, dim=-1)
    outputs: list[torch.Tensor] = []
    for step in range(time):
        q_t = q[:, step].unsqueeze(-1)
        k_t = k[:, step].unsqueeze(-1)
        v_t = v[:, step].unsqueeze(-1)
        alpha_t = alpha[:, step].unsqueeze(-1)
        beta_t = beta[:, step].unsqueeze(-1)
        error = v_t - state.matmul(k_t)
        state = alpha_t * state + beta_t * error.matmul(k_t.transpose(-1, -2))
        outputs.append(state.matmul(q_t).squeeze(-1))
    output = torch.stack(outputs, dim=1)
    return (output * g if g is not None else output), state


class GatedDeltaNetAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_head: int, variant: str = "scalar"):
        super().__init__()
        if variant not in {"scalar", "channel-wise"}:
            raise ValueError(f"Unknown Gated DeltaNet variant {variant!r}")
        self.d_model, self.n_head, self.d_head, self.variant = d_model, n_head, d_head, variant
        projection_dim = n_head * d_head
        self.q_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.k_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.v_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.q_conv = CausalConv1d(projection_dim)
        self.k_conv = CausalConv1d(projection_dim)
        self.v_conv = CausalConv1d(projection_dim)
        self.beta_proj = nn.Linear(d_model, n_head, bias=False)
        alpha_dim = n_head if variant == "scalar" else projection_dim
        self.alpha_proj = nn.Linear(d_model, alpha_dim, bias=False)
        self.gate_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.out_proj = nn.Linear(projection_dim, d_model, bias=False)
        self.layer_idx = -1
        self._gate_observer: Callable[[torch.Tensor, str, int], None] | None = None

    def set_gate_observer(self, observer: Callable[[torch.Tensor, str, int], None] | None) -> None:
        self._gate_observer = observer

    def forward(
        self, x: torch.Tensor, cache: Mapping[str, torch.Tensor | str] | None = None, *, use_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LayerCache]:
        batch, time, _ = x.shape
        heads, width = self.n_head, self.d_head
        cache = dict(cache or {})
        if use_cache:
            q, q_cache = self.q_conv(self.q_proj(x), cache.get("q_conv"), return_cache=True)
            k, k_cache = self.k_conv(self.k_proj(x), cache.get("k_conv"), return_cache=True)
            v, v_cache = self.v_conv(self.v_proj(x), cache.get("v_conv"), return_cache=True)
        else:
            q, k, v = self.q_conv(self.q_proj(x)), self.k_conv(self.k_proj(x)), self.v_conv(self.v_proj(x))
        q, k, v = (tensor.view(batch, time, heads, width) for tensor in (q, k, v))
        beta = torch.sigmoid(self.beta_proj(x)).view(batch, time, heads, 1)
        alpha_width = 1 if self.variant == "scalar" else width
        alpha = torch.sigmoid(self.alpha_proj(x)).view(batch, time, heads, alpha_width)
        if self._gate_observer is not None:
            self._gate_observer(alpha.detach(), self.variant, self.layer_idx)
        g = F.silu(self.gate_proj(x)).view(batch, time, heads, width)
        output, state = recurrent_forward(q, k, v, alpha, beta, g, cache.get("state"))
        output = self.out_proj(output.reshape(batch, time, heads * width))
        if not use_cache:
            return output
        return output, {
            "kind": "gdn",
            "state": state,
            "q_conv": q_cache,
            "k_conv": k_cache,
            "v_conv": v_cache,
        }


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_head: int):
        super().__init__()
        self.d_model, self.n_head, self.d_head = d_model, n_head, d_head
        projection_dim = n_head * d_head
        self.q_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.k_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.v_proj = nn.Linear(d_model, projection_dim, bias=False)
        self.out_proj = nn.Linear(projection_dim, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cache: Mapping[str, torch.Tensor | str] | None = None, *, use_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LayerCache]:
        batch, time, _ = x.shape
        heads, width = self.n_head, self.d_head
        q = self.q_proj(x).view(batch, time, heads, width).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, heads, width).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, heads, width).transpose(1, 2)
        cache = dict(cache or {})
        prior_k, prior_v = cache.get("k"), cache.get("v")
        if prior_k is not None or prior_v is not None:
            if not isinstance(prior_k, torch.Tensor) or not isinstance(prior_v, torch.Tensor):
                raise ValueError("Attention cache must contain K and V tensors together")
            if prior_k.shape[:2] != (batch, heads) or prior_k.shape != prior_v.shape:
                raise ValueError("Invalid attention K/V cache shape")
            if prior_k.device != k.device or prior_k.dtype != k.dtype:
                raise ValueError("Attention cache must match projected K/V device and dtype")
            previous_length = prior_k.size(2)
            all_k, all_v = torch.cat((prior_k, k), dim=2), torch.cat((prior_v, v), dim=2)
            query_positions = torch.arange(time, device=x.device)[:, None]
            key_positions = torch.arange(previous_length + time, device=x.device)[None, :]
            causal_mask = key_positions <= (previous_length + query_positions)
            output = F.scaled_dot_product_attention(q, all_k, all_v, attn_mask=causal_mask, is_causal=False)
        else:
            all_k, all_v = k, v
            output = F.scaled_dot_product_attention(q, all_k, all_v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(batch, time, heads * width)
        output = self.out_proj(output)
        return (output, {"kind": "attention", "k": all_k, "v": all_v}) if use_cache else output


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GatedDeltaNetBlock(nn.Module):
    def __init__(
        self, layer_idx: int, d_model: int, n_head: int, d_head: int, d_ff: int,
        variant: str, attention_layout: str,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm, self.mlp_norm = RMSNorm(d_model), RMSNorm(d_model)
        is_full_attention = attention_layout == "hybrid" and layer_idx % 4 == 3
        self.attn: GatedDeltaNetAttention | CausalSelfAttention = (
            CausalSelfAttention(d_model, n_head, d_head)
            if is_full_attention else GatedDeltaNetAttention(d_model, n_head, d_head, variant)
        )
        if isinstance(self.attn, GatedDeltaNetAttention):
            self.attn.layer_idx = layer_idx
        self.mlp = SwiGLUMLP(d_model, d_ff)

    def forward(
        self, x: torch.Tensor, cache: Mapping[str, torch.Tensor | str] | None = None, *, use_cache: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, LayerCache]:
        attn_result = self.attn(self.attn_norm(x), cache, use_cache=use_cache)
        if use_cache:
            attn_out, new_cache = attn_result  # type: ignore[misc]
        else:
            attn_out = attn_result  # type: ignore[assignment]
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return (x, new_cache) if use_cache else x


def _normalize_config(config_dict: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    if isinstance(config_dict, ModelConfig):
        return config_dict
    raw = asdict(config_dict) if is_dataclass(config_dict) else dict(config_dict)
    raw.setdefault("name", "custom")
    raw.setdefault("attention_layout", "hybrid")
    d_ff = raw["d_ff"]
    raw["d_ff"] = tuple(d_ff) if isinstance(d_ff, list) else d_ff
    fields = ("name", "d_model", "n_layer", "n_head", "d_ff", "attention_layout")
    return ModelConfig(**{key: raw[key] for key in fields})


class GatedDeltaNetLM(nn.Module):
    def __init__(self, config_dict: ModelConfig | Mapping[str, Any], variant: str = "scalar", vocab_size: int = 50_257):
        super().__init__()
        if variant == "scalar-matched":
            variant = "scalar"
        self.model_config = _normalize_config(config_dict)
        self.variant, self.vocab_size, self.d_model = variant, vocab_size, self.model_config.d_model
        self.tok_embeddings = nn.Embedding(vocab_size, self.d_model)
        self.blocks = nn.ModuleList([
            GatedDeltaNetBlock(
                index, self.d_model, self.model_config.n_head, self.model_config.d_head,
                self.model_config.ffn_widths[index], variant, self.model_config.attention_layout,
            )
            for index in range(self.model_config.n_layer)
        ])
        self.norm = RMSNorm(self.d_model)
        self.output = nn.Linear(self.d_model, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def set_gate_observer(self, observer: Callable[[torch.Tensor, str, int], None] | None) -> None:
        for block in self.blocks:
            if isinstance(block.attn, GatedDeltaNetAttention):
                block.attn.set_gate_observer(observer)

    def _validate_model_cache(self, cache: Sequence[Mapping[str, torch.Tensor | str]] | None) -> None:
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError(f"Expected {len(self.blocks)} layer caches, got {len(cache)}")

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None,
        cache: Sequence[Mapping[str, torch.Tensor | str]] | None = None, *, use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[LayerCache]]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, time)")
        self._validate_model_cache(cache)
        hidden = self.tok_embeddings(tokens)
        new_cache: list[LayerCache] = []
        for index, block in enumerate(self.blocks):
            result = block(hidden, None if cache is None else cache[index], use_cache=use_cache)
            if use_cache:
                hidden, layer_cache = result  # type: ignore[misc]
                new_cache.append(layer_cache)
            else:
                hidden = result  # type: ignore[assignment]
        logits = self.output(self.norm(hidden))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        return (logits, loss, new_cache) if use_cache else (logits, loss)

    @torch.no_grad()
    def prefill(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list[LayerCache]]:
        logits, _, cache = self(tokens, use_cache=True)
        return logits, cache

    @torch.no_grad()
    def decode(self, token: torch.Tensor, cache: Sequence[Mapping[str, torch.Tensor | str]]) -> tuple[torch.Tensor, list[LayerCache]]:
        if token.ndim != 2 or token.size(1) != 1:
            raise ValueError("decode accepts exactly one token per batch item")
        logits, _, next_cache = self(token, cache=cache, use_cache=True)
        return logits, next_cache

    @torch.no_grad()
    def greedy_decode(self, prompt: torch.Tensor, generated_tokens: int) -> tuple[torch.Tensor, list[LayerCache]]:
        logits, cache = self.prefill(prompt)
        output = prompt
        next_token = logits[:, -1:].argmax(dim=-1)
        for step in range(generated_tokens):
            output = torch.cat((output, next_token), dim=1)
            if step + 1 < generated_tokens:
                logits, cache = self.decode(next_token, cache)
                next_token = logits[:, -1:].argmax(dim=-1)
        return output, cache


def cache_num_bytes(cache: Sequence[Mapping[str, torch.Tensor | str]]) -> int:
    """Report cache tensor storage once, even if a tensor is referenced twice."""

    seen: set[int] = set()
    total = 0
    for layer in cache:
        for value in layer.values():
            if isinstance(value, torch.Tensor) and value.data_ptr() not in seen:
                seen.add(value.data_ptr())
                total += value.numel() * value.element_size()
    return total


def parameter_breakdown(model: GatedDeltaNetLM) -> dict[str, int]:
    """A non-overlapping breakdown of trainable parameter storage."""

    categories = {
        "embeddings": 0, "gates": 0, "projections": 0, "convolutions": 0,
        "ffns": 0, "norms": 0, "output_head": 0, "other": 0,
    }
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if name.startswith("tok_embeddings"):
            categories["embeddings"] += count
        elif ".mlp." in name:
            categories["ffns"] += count
        elif ".norm" in name or name.startswith("norm."):
            categories["norms"] += count
        elif ".conv." in name:
            categories["convolutions"] += count
        elif any(part in name for part in ("alpha_proj", "beta_proj", "gate_proj")):
            categories["gates"] += count
        elif any(part in name for part in ("q_proj", "k_proj", "v_proj", "out_proj")):
            categories["projections"] += count
        elif name.startswith("output"):
            categories["output_head"] += count
        else:
            categories["other"] += count
    categories["total"] = sum(categories.values())
    categories["tied_output_head"] = 0
    return categories
