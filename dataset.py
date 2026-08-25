"""Deterministic TinyStories and offline toy-language data loaders."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from artifacts import atomic_write_json, read_json
from config import ExperimentConfig


class ToyLanguageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A tiny local dataset for smoke tests; no Hugging Face or tokenizer access."""

    def __init__(self, *, examples: int, context_length: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        # A predictable local pattern gives the two-step smoke path a valid LM loss
        # without pretending this represents TinyStories quality.
        starts = torch.randint(0, vocab_size, (examples, 1), generator=generator)
        offsets = torch.arange(context_length + 1).view(1, -1)
        self.tokens = (starts + offsets) % vocab_size

    def __len__(self) -> int:
        return self.tokens.size(0)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.tokens[index]
        return sequence[:-1].clone(), sequence[1:].clone()


def toy_dataloader(profile: ExperimentConfig, *, seed: int, split: str) -> DataLoader:
    examples = 12 if split == "train" else 4
    dataset = ToyLanguageDataset(
        examples=examples, context_length=profile.budget.context_length,
        vocab_size=profile.vocab_size, seed=seed + (0 if split == "train" else 10_000),
    )
    return DataLoader(dataset, batch_size=profile.budget.batch_size, shuffle=False, num_workers=0)


def split_manifest_path(profile: ExperimentConfig, *, seed: int, split: str) -> Path:
    return profile.output_dir / "data_manifests" / f"tinystories_seed{seed}_{split}.json"


def _dataset_identity(dataset: object) -> str:
    return str(getattr(dataset, "_fingerprint", "unknown"))


def load_tinystories_dataset(
    profile: ExperimentConfig, *, split: str, data_seed: int, run_dir: str | Path | None = None
) -> IterableDataset:
    """Load/cache the production corpus with a persisted deterministic order identity."""

    if profile.run_mode == "smoke":
        raise RuntimeError("Smoke mode must use toy_dataloader and never access TinyStories")
    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("TinyStories requires the datasets and tiktoken dependencies") from exc

    cache_dir = Path("data") / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("roneneldan/TinyStories", split=split, cache_dir=str(cache_dir))
    if split == "train":
        dataset = dataset.shuffle(seed=data_seed)
    identity = {
        "dataset_name": "roneneldan/TinyStories",
        "split": split,
        "data_seed": data_seed,
        "shuffle_algorithm": "datasets.Dataset.shuffle",
        "dataset_fingerprint": _dataset_identity(dataset),
        "context_length": profile.budget.context_length,
    }
    identity["identity_hash"] = sha256(str(sorted(identity.items())).encode()).hexdigest()
    target = split_manifest_path(profile, seed=data_seed, split=split)
    if target.exists():
        existing = read_json(target)
        if existing != identity:
            raise RuntimeError("TinyStories split manifest drifted; refusing to change paired data order")
    else:
        atomic_write_json(target, identity)
    if run_dir is not None:
        atomic_write_json(Path(run_dir) / "data_identity.json", identity)
    return TinyStoriesDataset(dataset, tiktoken.get_encoding("gpt2"), profile.budget.context_length)


class TinyStoriesDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: object, tokenizer: object, context_length: int):
        super().__init__()
        self.dataset, self.tokenizer, self.context_length = dataset, tokenizer, context_length

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        token_buffer: list[int] = []
        for story in self.dataset:  # type: ignore[union-attr]
            text = story.get("text", "")
            if not text:
                continue
            token_buffer.extend(self.tokenizer.encode(text, allowed_special="all"))  # type: ignore[union-attr]
            token_buffer.append(self.tokenizer.eot_token)  # type: ignore[union-attr]
            while len(token_buffer) >= self.context_length + 1:
                chunk = token_buffer[: self.context_length + 1]
                token_buffer = token_buffer[self.context_length :]
                yield torch.tensor(chunk[:-1]), torch.tensor(chunk[1:])


def tinystories_dataloader(
    profile: ExperimentConfig, *, split: str, data_seed: int, run_dir: str | Path | None = None
) -> DataLoader:
    dataset = load_tinystories_dataset(profile, split=split, data_seed=data_seed, run_dir=run_dir)
    return DataLoader(dataset, batch_size=profile.budget.batch_size, num_workers=0)
