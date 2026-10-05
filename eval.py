"""Cache-aware inference benchmarks and persisted exploratory needle prompts."""

from __future__ import annotations

from hashlib import sha256
import random
import time
from pathlib import Path
from typing import Any

import torch

from artifacts import atomic_write_json, read_json
from config import ExperimentConfig
from energy import EnergyProvider, normalized_energy
from model import GatedDeltaNetLM, cache_num_bytes


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


@torch.no_grad()
def cache_equivalence_error(model: GatedDeltaNetLM, tokens: torch.Tensor) -> float:
    """Maximum full-vs-cached logit difference for the supplied deterministic case."""

    model.eval()
    full_logits, _ = model(tokens)
    first_logits, cache = model.prefill(tokens[:, :1])
    pieces = [first_logits]
    for position in range(1, tokens.size(1)):
        logits, cache = model.decode(tokens[:, position : position + 1], cache)
        pieces.append(logits)
    return float((full_logits - torch.cat(pieces, dim=1)).abs().max().item())


@torch.no_grad()
def benchmark_inference(
    model: GatedDeltaNetLM,
    profile: ExperimentConfig,
    device: torch.device,
    energy_provider: EnergyProvider,
    *,
    prompt_lengths: tuple[int, ...] | None = None,
    generated_tokens: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure optional full-prefix generation, prefill, and cached decode separately."""

    model.eval()
    model.set_gate_observer(None)  # Diagnostics must never pollute timed regions.
    prompt_lengths = prompt_lengths or ((4, 8) if profile.run_mode == "smoke" else (64, 128, 256, 512))
    generated_tokens = generated_tokens or (2 if profile.run_mode == "smoke" else profile.budget.context_length)
    generator = torch.Generator(device=device).manual_seed(seed)
    results: dict[str, Any] = {}
    for prompt_length in prompt_lengths:
        prompt = torch.randint(0, profile.vocab_size, (1, prompt_length), generator=generator, device=device)
        for _ in range(profile.benchmark_warmup):
            logits, cache = model.prefill(prompt)
            model.decode(logits[:, -1:].argmax(dim=-1), cache)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        full_prefix_result: dict[str, Any]
        if profile.enable_full_prefix_benchmark:
            full_times: list[float] = []
            with energy_provider.region("full_prefix_generation") as full_energy:
                for _ in range(profile.benchmark_repeats):
                    current = prompt.clone()
                    _sync(device)
                    started = time.perf_counter()
                    for _ in range(generated_tokens):
                        logits, _ = model(current)
                        current = torch.cat((current, logits[:, -1:].argmax(dim=-1)), dim=1)
                    _sync(device)
                    full_times.append(time.perf_counter() - started)
            assert full_energy.result is not None
            full_average = sum(full_times) / len(full_times)
            full_prefix_result = {
                "enabled": True,
                "latency_seconds": full_average,
                "milliseconds_per_generated_token": full_average / generated_tokens * 1000,
                "tokens_per_second": generated_tokens / full_average,
                "peak_memory_mb": _peak_memory_mb(device),
                "energy": normalized_energy(
                    full_energy.result,
                    tokens=generated_tokens * profile.benchmark_repeats,
                    unit="generated_token",
                ),
            }
        else:
            full_prefix_result = {"enabled": False, "reason": "disabled in experiment profile"}

        prefill_times: list[float] = []
        prefill_cache_bytes: list[int] = []
        with energy_provider.region("prefill") as prefill_energy:
            for _ in range(profile.benchmark_repeats):
                _sync(device)
                started = time.perf_counter()
                _, cache = model.prefill(prompt)
                _sync(device)
                prefill_times.append(time.perf_counter() - started)
                prefill_cache_bytes.append(cache_num_bytes(cache))

        decode_times: list[float] = []
        with energy_provider.region("cached_decode") as decode_energy:
            for _ in range(profile.benchmark_repeats):
                logits, cache = model.prefill(prompt)
                token = logits[:, -1:].argmax(dim=-1)
                _sync(device)
                started = time.perf_counter()
                for _ in range(generated_tokens):
                    logits, cache = model.decode(token, cache)
                    token = logits[:, -1:].argmax(dim=-1)
                _sync(device)
                decode_times.append(time.perf_counter() - started)

        assert prefill_energy.result and decode_energy.result
        prefix = f"context_{prompt_length}"
        prefill_average = sum(prefill_times) / len(prefill_times)
        decode_average = sum(decode_times) / len(decode_times)
        results[prefix] = {
            "metadata": {
                "prompt_length": prompt_length, "generated_tokens": generated_tokens, "batch_size": 1,
                "dtype": str(next(model.parameters()).dtype), "warmup": profile.benchmark_warmup,
                "repetitions": profile.benchmark_repeats, "synchronized": device.type == "cuda",
            },
            "full_prefix_generation": full_prefix_result,
            "prefill": {
                "latency_seconds": prefill_average, "time_to_first_token_seconds": prefill_average,
                "tokens_per_second": prompt_length / prefill_average,
                "cache_bytes": max(prefill_cache_bytes), "peak_memory_mb": _peak_memory_mb(device),
                "energy": normalized_energy(prefill_energy.result, tokens=prompt_length * profile.benchmark_repeats, unit="prefill_token"),
            },
            "cached_decode": {
                "latency_seconds": decode_average, "milliseconds_per_generated_token": decode_average / generated_tokens * 1000,
                "tokens_per_second": generated_tokens / decode_average,
                "cache_bytes": cache_num_bytes(cache), "peak_memory_mb": _peak_memory_mb(device),
                "energy": normalized_energy(decode_energy.result, tokens=generated_tokens * profile.benchmark_repeats, unit="generated_token"),
            },
        }
    return results


def persisted_needle_suite(path: str | Path, *, seed: int, cases: int = 10) -> dict[str, Any]:
    """Persist paired natural-language diagnostics; this is explicitly exploratory."""

    target = Path(path)
    if target.exists():
        return read_json(target)
    rng = random.Random(seed)
    colors = ("blue", "green", "red", "pink", "gold")
    distractors = (
        "The small green frog sat on a big leaf in the pond.",
        "A shiny red car drove fast down the bumpy road.",
        "She picked a yellow flower and smelled its sweet scent.",
        "The blue sky was covered in fluffy white clouds today.",
    )
    prompts = []
    for index in range(cases):
        color = rng.choice(colors)
        before = " ".join(rng.choice(distractors) for _ in range(2 + index % 3))
        after = " ".join(rng.choice(distractors) for _ in range(2))
        prompts.append({
            "id": f"needle-{index}", "prompt": f"{before} The secret key is {color}. {after} What is the secret key? The secret key is",
            "target": f" {color}", "depth_group": index % 3,
        })
    suite = {"version": "needle-suite-v1", "seed": seed, "label": "exploratory_zero_shot", "cases": prompts}
    suite["hash"] = sha256(str(suite).encode()).hexdigest()
    atomic_write_json(target, suite)
    return suite


@torch.no_grad()
def evaluate_needle_suite(model: GatedDeltaNetLM, suite: dict[str, Any], device: torch.device) -> dict[str, Any]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("Needle diagnostics require tiktoken") from exc
    tokenizer = tiktoken.get_encoding("gpt2")
    correct = 0
    rows = []
    model.eval()
    for case in suite["cases"]:
        tokens = torch.tensor([tokenizer.encode(case["prompt"])], device=device)
        logits, _ = model(tokens)
        predicted = logits[:, -1].argmax(dim=-1).item()
        target = tokenizer.encode(case["target"])[0]
        is_correct = predicted == target
        correct += is_correct
        rows.append({"id": case["id"], "correct": is_correct, "depth_group": case["depth_group"]})
    return {"label": "exploratory_zero_shot", "suite_hash": suite["hash"], "accuracy": correct / len(rows), "cases": rows}
