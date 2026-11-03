"""Fixed-budget, deterministic training used by every experiment family."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.optim as optim

from artifacts import save_checkpoint
from config import ExperimentConfig
from energy import EnergyProvider, normalized_energy
from model import GatedDeltaNetLM


PROGRESS_INTERVAL = 50


@dataclass(frozen=True)
class SeedBundle:
    research_seed: int
    model_seed: int
    data_seed: int
    evaluation_seed: int

    @classmethod
    def from_research_seed(cls, seed: int) -> "SeedBundle":
        return cls(seed, seed, seed + 100_000, seed + 200_000)

    def to_dict(self) -> dict[str, int]:
        return {
            "research_seed": self.research_seed,
            "model_seed": self.model_seed,
            "data_seed": self.data_seed,
            "evaluation_seed": self.evaluation_seed,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The study values reproducibility more than benchmarking an unrecorded
    # implementation choice. CUDA benchmarking is intentionally not enabled.
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_model(
    model: GatedDeltaNetLM, dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device, max_steps: int
) -> dict[str, float | int]:
    model_was_training = model.training
    model.eval()
    iterator = iter(dataloader)
    losses: list[float] = []
    for _ in range(max_steps):
        try:
            tokens, targets = next(iterator)
        except StopIteration:
            break
        _, loss = model(tokens.to(device), targets.to(device))
        assert loss is not None
        losses.append(loss.item())
    model.train(model_was_training)
    if not losses:
        raise RuntimeError("Validation data produced no batches")
    loss = sum(losses) / len(losses)
    return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss)).item()), "steps": len(losses)}


def _next_batch(
    iterator: Any, dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[tuple[torch.Tensor, torch.Tensor], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(dataloader)
        try:
            return next(iterator), iterator
        except StopIteration as exc:
            raise RuntimeError("Training data produced no batches") from exc


def train_fixed_budget(
    *,
    model: GatedDeltaNetLM,
    train_dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validation_dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    profile: ExperimentConfig,
    device: torch.device,
    run_dir: str,
    energy_provider: EnergyProvider,
    progress_label: str = "training",
) -> dict[str, Any]:
    """Train to the exact configured step budget; validation can never stop it."""

    profile.budget.require_ready()
    assert profile.budget.max_optimizer_steps is not None
    max_steps = profile.budget.max_optimizer_steps
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=profile.learning_rate, weight_decay=profile.weight_decay,
        betas=(profile.beta1, profile.beta2),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=profile.learning_rate, total_steps=max_steps, pct_start=0.05,
        anneal_strategy="cos", div_factor=10.0, final_div_factor=10.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    iterator = iter(train_dataloader)
    losses: list[float] = []
    unclipped_norms: list[float] = []
    clipped_norms: list[float] = []
    step_durations: list[float] = []
    events: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_validation_metrics: dict[str, Any] | None = None
    processed_tokens = microbatches = 0
    best_validation = float("inf")
    start = time.perf_counter()

    with energy_provider.region("training") as energy_region:
        model.train()
        for step in range(1, max_steps + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(profile.budget.grad_accum_steps):
                (tokens, targets), iterator = _next_batch(iterator, train_dataloader)
                tokens, targets = tokens.to(device), targets.to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    _, loss = model(tokens, targets)
                    assert loss is not None
                    if not torch.isfinite(loss):
                        events.append({"step": step, "kind": "nonfinite_loss"})
                        raise FloatingPointError(f"Non-finite loss at optimizer step {step}")
                    scaled_loss = loss / profile.budget.grad_accum_steps
                scaler.scale(scaled_loss).backward()
                loss_sum += loss.item()
                processed_tokens += tokens.numel()
                microbatches += 1
            scaler.unscale_(optimizer)
            unclipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), profile.grad_clip).item())
            unclipped_norms.append(unclipped)
            clipped_norms.append(min(unclipped, profile.grad_clip))
            if not math_isfinite(unclipped):
                events.append({"step": step, "kind": "nonfinite_gradient"})
                raise FloatingPointError(f"Non-finite gradient norm at optimizer step {step}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss_sum)
            if len(losses) >= 6 and loss_sum > 5 * (sum(losses[-6:-1]) / 5):
                events.append({"step": step, "kind": "loss_spike", "loss": loss_sum})
            step_durations.append(time.perf_counter() - step_started)

            if step % PROGRESS_INTERVAL == 0 or step == max_steps:
                print(
                    f"Training progress [{progress_label}]: {step}/{max_steps} optimizer steps completed",
                    flush=True,
                )

            if step % profile.budget.evaluation_interval == 0 or step == max_steps:
                validation = evaluate_model(model, validation_dataloader, device, profile.budget.validation_steps)
                validation["step"] = step
                validation_history.append(validation)
                if float(validation["loss"]) < best_validation:
                    best_validation = float(validation["loss"])
                    best_validation_metrics = dict(validation)
                    torch.save({"model_state": model.state_dict(), "step": step, "validation": validation}, f"{run_dir}/best_validation_checkpoint.pt")

    wall_time = time.perf_counter() - start
    checkpoint = save_checkpoint(
        f"{run_dir}/final_checkpoint.pt", model=model, optimizer=optimizer,
        completed_steps=max_steps, processed_tokens=processed_tokens,
    )
    primary_validation = evaluate_model(model, validation_dataloader, device, profile.budget.validation_steps)
    assert energy_region.result is not None
    energy = normalized_energy(energy_region.result, tokens=processed_tokens / 1_000_000, unit="million_training_tokens")
    return {
        "training": {
            "fixed_optimizer_steps": max_steps,
            "processed_tokens": processed_tokens,
            "microbatches": microbatches,
            "wall_time_seconds": wall_time,
            "optimizer_step_seconds": step_durations,
            "training_tokens_per_second": processed_tokens / wall_time if wall_time else None,
            "train_losses": losses,
            "unclipped_gradient_norms": unclipped_norms,
            "clipped_gradient_norms": clipped_norms,
            "stability_events": events,
        },
        "final_checkpoint": checkpoint,
        "final_validation": primary_validation,
        "best_validation_secondary": best_validation_metrics,
        "validation_history": validation_history,
        "training_energy": energy,
    }


def math_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))
