"""Full-backbone fine-tuning loop for dynamic GDN decision models."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from decision_model import GDNDecisionModel
from decision_research_config import TrainingConfig
from decision_artifacts import (
    ResumeCheckpointStore,
    build_resume_payload,
    restore_rng_state,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loss_for_examples(model: GDNDecisionModel, examples: Sequence[dict[str, Any]], device: torch.device):
    logits_list = []
    targets = []
    for example in examples:
        ids = example["input_ids"].to(device)
        logits = model.score(ids, int(example["decision_index"]), example["option_spans"])
        target = torch.as_tensor(example["target"], dtype=torch.float32, device=device)
        if logits.numel() != target.numel():
            raise ValueError("Candidate logits and target lengths differ")
        if not torch.isfinite(target).all() or (target < 0).any() or not torch.isclose(
            target.sum(), target.new_tensor(1.0), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("Target distribution must be finite, nonnegative, and normalized")
        logits_list.append(logits.float())
        targets.append(target)
    losses = [-(target * F.log_softmax(logits, dim=-1)).sum() for logits, target in zip(logits_list, targets)]
    return torch.stack(losses).mean(), logits_list


def _permute_options(example: dict[str, Any], generator: torch.Generator) -> dict[str, Any]:
    """Permute contiguous option spans and remap soft targets in lockstep."""
    import copy

    spans = [tuple(map(int, span)) for span in example["option_spans"]]
    if len(spans) < 2:
        return example
    if any(spans[index][1] != spans[index + 1][0] for index in range(len(spans) - 1)):
        raise ValueError("Training option spans must be contiguous for order augmentation")
    order = torch.randperm(len(spans), generator=generator).tolist()
    ids = torch.as_tensor(example["input_ids"], dtype=torch.long)
    chunks = [ids[start:end] for start, end in spans]
    prefix, suffix = ids[:spans[0][0]], ids[spans[-1][1]:]
    pieces = [prefix]
    new_spans = []
    cursor = len(prefix)
    for index in order:
        chunk = chunks[index]
        pieces.append(chunk)
        new_spans.append((cursor, cursor + len(chunk)))
        cursor += len(chunk)
    pieces.append(suffix)
    result = copy.copy(example)
    result["input_ids"] = torch.cat(pieces)
    result["option_spans"] = new_spans
    result["target"] = torch.as_tensor(example["target"])[order]
    if "serialized_labels" in example:
        result["serialized_labels"] = [example["serialized_labels"][index] for index in order]
    prior_order = list(example.get("option_order", range(len(spans))))
    result["option_order"] = [prior_order[index] for index in order]
    return result


@torch.inference_mode()
def validation_soft_nll(
    model: GDNDecisionModel,
    examples: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    if not examples:
        raise ValueError("Validation set is empty")
    model.eval()
    total_loss = 0.0
    total_count = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        loss, _ = _loss_for_examples(model, batch, device)
        total_loss += float(loss) * len(batch)
        total_count += len(batch)
    return total_loss / total_count


def train_variant(
    model: GDNDecisionModel,
    train_examples: Sequence[dict[str, Any]],
    validation_examples: Sequence[dict[str, Any]],
    *,
    config: TrainingConfig,
    backbone_learning_rate: float,
    seed: int,
    output_dir: str | Path,
    run_fingerprint: str,
    device: torch.device | None = None,
    checkpoint_callback=None,
    validation_callback=None,
    resume_store: ResumeCheckpointStore | None = None,
    resume_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one selected model and emit best/final inference checkpoints.

    Callbacks are hooks for the run lifecycle/checkpoint service. Resumable
    state is intentionally delegated to that service rather than silently
    approximated inside the optimizer loop.
    """
    if not train_examples or not validation_examples:
        raise ValueError("Training and validation examples must be non-empty")
    if backbone_learning_rate <= 0:
        raise ValueError("backbone_learning_rate must be positive")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    model.to(device)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_learning_rate},
            {"params": head_params, "lr": config.head_learning_rate},
        ],
        weight_decay=config.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_examples) / config.physical_batch_size)
    updates_per_epoch = math.ceil(steps_per_epoch / config.gradient_accumulation_steps)
    total_updates = updates_per_epoch * config.max_epochs
    warmup_updates = int(total_updates * config.warmup_fraction)

    def lr_multiplier(step: int) -> float:
        if warmup_updates and step < warmup_updates:
            return max((step + 1) / warmup_updates, 1e-8)
        progress = (step - warmup_updates) / max(total_updates - warmup_updates, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    amp_enabled = device.type == "cuda" and config.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "training_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    best_nll = float("inf")
    best_step = 0
    global_step = 0
    resume_epoch = 0
    resume_microbatch_index = 0
    resume_epoch_order: list[int] | None = None
    if resume_payload is not None:
        model_state = resume_payload.get("model_state")
        head_state = resume_payload.get("head_state")
        if not isinstance(model_state, dict) or not isinstance(head_state, dict):
            raise ValueError("Resume checkpoint is missing backbone/head state")
        model.backbone.load_state_dict(model_state, strict=True)
        model.head.load_state_dict(head_state, strict=True)
        optimizer_state = resume_payload.get("optimizer_state")
        scheduler_state = resume_payload.get("scheduler_state")
        scaler_state = resume_payload.get("scaler_state")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        if isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        resume_epoch = int(resume_payload.get("epoch", 0))
        global_step = int(resume_payload.get("completed_optimizer_step", 0))
        best_state = resume_payload.get("best_validation_state", {})
        if isinstance(best_state, dict):
            best_nll = float(best_state.get("validation_nll", best_nll))
            best_step = int(best_state.get("step", best_step))
        sampler_state = resume_payload.get("sampler_state", {})
        if isinstance(sampler_state, dict):
            resume_microbatch_index = int(sampler_state.get("next_microbatch_index", 0))
            stored_order = sampler_state.get("epoch_order")
            if stored_order is not None:
                resume_epoch_order = [int(index) for index in stored_order]
            generator_state = sampler_state.get("generator_state")
            if isinstance(generator_state, torch.Tensor):
                generator.set_state(generator_state)
        rng_state = resume_payload.get("rng_state")
        if isinstance(rng_state, Mapping):
            restore_rng_state(rng_state)
    total_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(resume_epoch, config.max_epochs):
        if epoch == resume_epoch and resume_epoch_order is not None:
            order = resume_epoch_order
        else:
            order = torch.randperm(len(train_examples), generator=generator).tolist()
        microbatches = [
            order[index : index + config.physical_batch_size]
            for index in range(0, len(order), config.physical_batch_size)
        ]
        start_microbatch = resume_microbatch_index if epoch == resume_epoch else 0
        for group_start in range(start_microbatch, len(microbatches), config.gradient_accumulation_steps):
            group = microbatches[group_start : group_start + config.gradient_accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            summed_loss = 0.0
            item_count = 0
            for indices in group:
                examples = [_permute_options(train_examples[index], generator) for index in indices]
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    loss, _ = _loss_for_examples(model, examples, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, step={global_step + 1}")
                scaler.scale(loss * (len(examples) / sum(len(mb) for mb in group))).backward()
                summed_loss += float(loss.detach()) * len(examples)
                item_count += len(examples)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise FloatingPointError(f"Non-finite gradient norm at epoch={epoch + 1}, step={global_step + 1}")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            train_nll = summed_loss / max(item_count, 1)
            event: dict[str, Any] = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_soft_nll": train_nll,
                "gradient_norm": float(grad_norm),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
                "elapsed_seconds": time.perf_counter() - total_started,
            }
            if global_step % config.validation_every_steps == 0 or group_start + len(group) == len(microbatches):
                val_nll = validation_soft_nll(
                    model, validation_examples, batch_size=config.physical_batch_size, device=device
                )
                event["validation_soft_nll"] = val_nll
                if validation_callback is not None:
                    validation_callback(global_step, val_nll, model)
                if val_nll < best_nll:
                    best_nll, best_step = val_nll, global_step
                    model.save_inference_checkpoint(
                        output / "best_model.pt", run_fingerprint=run_fingerprint, selected_step=global_step,
                        validation_nll=val_nll,
                    )
                model.train()
            with history_path.open("a", encoding="utf-8") as history:
                history.write(json.dumps(event, sort_keys=True) + "\n")
            at_epoch_end = group_start + len(group) == len(microbatches)
            if checkpoint_callback is not None and (
                global_step % config.checkpoint_every_steps == 0 or at_epoch_end
            ):
                checkpoint_callback(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    step=global_step,
                    sampler_position=group_start + len(group),
                    best_validation_nll=best_nll,
                    best_step=best_step,
                    generator_state=generator.get_state(),
                    python_rng_state=random.getstate(),
                    numpy_rng_state=np.random.get_state(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                )
            if resume_store is not None and (
                global_step % config.checkpoint_every_steps == 0 or at_epoch_end
            ):
                next_epoch = epoch + 1 if at_epoch_end else epoch
                sampler_state = {
                    "epoch_order": None if at_epoch_end else order,
                    "next_microbatch_index": 0 if at_epoch_end else group_start + len(group),
                    "generator_state": generator.get_state(),
                }
                payload = build_resume_payload(
                    model=model.backbone,
                    head=model.head,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=next_epoch,
                    completed_optimizer_step=global_step,
                    sampler_state=sampler_state,
                    best_validation_state={"validation_nll": best_nll, "step": best_step},
                    run_fingerprint=run_fingerprint,
                    extra={"variant": model.variant, "seed": seed},
                )
                resume_store.save(payload, run_fingerprint=run_fingerprint)
        resume_microbatch_index = 0
        resume_epoch_order = None

    final_nll = validation_soft_nll(
        model, validation_examples, batch_size=config.physical_batch_size, device=device
    )
    final_hash = model.save_inference_checkpoint(
        output / "final_model.pt", run_fingerprint=run_fingerprint, selected_step=global_step,
        validation_nll=final_nll, metadata={"kind": "fixed_budget_final"},
    )
    return {
        "variant": model.variant,
        "epochs": config.max_epochs,
        "optimizer_steps": global_step,
        "best_step": best_step,
        "best_validation_nll": best_nll,
        "final_validation_nll": final_nll,
        "final_checkpoint_sha256": final_hash,
        "elapsed_seconds": time.perf_counter() - total_started,
        "config": asdict(config),
    }
