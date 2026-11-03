"""Versioned deterministic MQAR and overwrite-recall data and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import random
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from artifacts import canonical_json
from config import RetrievalConfig
from model import GatedDeltaNetLM


GENERATOR_VERSION = "mqar-overwrite-v2"
QUERY_MARKER = 0
FILLER_TOKEN = 1


@dataclass(frozen=True)
class AnswerPosition:
    label_position: int
    target: int
    stale_values: tuple[int, ...]
    kind: Literal["latest", "unrelated", "mqar"]
    overwrite_count: int
    distance: int


@dataclass(frozen=True)
class RetrievalExample:
    tokens: torch.Tensor
    labels: torch.Tensor
    answers: tuple[AnswerPosition, ...]
    metadata: dict[str, int | str]


class RetrievalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Indexed, generated-on-demand data so the 100k production set stays compact."""

    def __init__(
        self, *, task: Literal["mqar", "overwrite"], examples: int, config: RetrievalConfig,
        seed: int, split: Literal["train", "held_out"], sequence_length: int | None = None,
    ):
        if config.vocabulary_size < 16:
            raise ValueError("Retrieval vocabulary must reserve enough symbolic keys and values")
        self.task, self.examples, self.config, self.seed, self.split = task, examples, config, seed, split
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.examples

    def _rng(self, index: int) -> random.Random:
        domain = 1 if self.split == "train" else 2
        return random.Random((self.seed << 24) ^ (domain << 20) ^ index)

    def _difficulty(self, index: int) -> tuple[int, int, int, int]:
        length = self.sequence_length or self.config.sequence_lengths[index % len(self.config.sequence_lengths)]
        key_count = self.config.key_counts[(index // len(self.config.sequence_lengths)) % len(self.config.key_counts)]
        distance = max(0, self.config.query_distances[(index // (len(self.config.sequence_lengths) * len(self.config.key_counts))) % len(self.config.query_distances)])
        overwrites = self.config.overwrite_counts[index % len(self.config.overwrite_counts)]
        # Reserve a marker and ensure small smoke sequences always remain valid.
        maximum_keys = max(1, (length - 6) // 2)
        return length, min(key_count, maximum_keys), min(distance, max(1, length - 6)), overwrites

    def example(self, index: int) -> RetrievalExample:
        rng = self._rng(index)
        length, key_count, distance, overwrite_count = self._difficulty(index)
        if self.task == "mqar":
            return self._mqar(rng, length, key_count, distance)
        return self._overwrite(rng, length, key_count, distance, overwrite_count)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.example(index)
        # Training batches need one tensor shape while held-out evaluation still
        # reads each unpadded example and preserves its true difficulty length.
        target_length = max(self.config.sequence_lengths) - 1
        if example.tokens.numel() == target_length:
            return example.tokens, example.labels
        padded_tokens = torch.full((target_length,), FILLER_TOKEN, dtype=torch.long)
        padded_labels = torch.full((target_length,), -100, dtype=torch.long)
        padded_tokens[: example.tokens.numel()] = example.tokens
        padded_labels[: example.labels.numel()] = example.labels
        return padded_tokens, padded_labels

    def _symbols(self, rng: random.Random, count: int) -> tuple[list[int], list[int]]:
        key_upper = self.config.vocabulary_size // 2
        keys = rng.sample(list(range(2, key_upper)), count)
        values = rng.sample(list(range(key_upper, self.config.vocabulary_size)), count)
        return keys, values

    @staticmethod
    def _labels(sequence: list[int], answers: list[AnswerPosition]) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.full_like(tokens, -100)
        for answer in answers:
            labels[answer.label_position] = answer.target
        return tokens, labels

    def _mqar(self, rng: random.Random, length: int, key_count: int, distance: int) -> RetrievalExample:
        keys, values = self._symbols(rng, key_count)
        memory = [symbol for pair in zip(keys, values) for symbol in pair]
        queried = list(range(key_count))
        # Define query distance as the number of filler tokens between the end of
        # the memory block and the first query marker. Preserve that gap exactly
        # when feasible; record the realized (possibly capacity-limited) gap.
        realized_distance = min(max(0, distance), max(0, length - len(memory) - 3))
        query_capacity = max(1, (length - len(memory) - realized_distance) // 3)
        queried = queried[:query_capacity]
        query_groups = [[QUERY_MARKER, keys[item], values[item]] for item in queried]
        remaining = max(0, length - len(memory) - realized_distance - 3 * len(query_groups))
        prefix_filler = [FILLER_TOKEN] * remaining
        gap_filler = [FILLER_TOKEN] * realized_distance
        sequence = prefix_filler + memory + gap_filler + [symbol for group in query_groups for symbol in group]
        sequence.extend([FILLER_TOKEN] * (length - len(sequence)))
        answers: list[AnswerPosition] = []
        first_query = len(prefix_filler) + len(memory) + realized_distance
        for item_index, item in enumerate(queried):
            query_start = first_query + 3 * item_index
            label_position = query_start + 1
            answers.append(AnswerPosition(label_position, values[item], (), "mqar", 0, realized_distance))
        tokens, labels = self._labels(sequence, answers)
        return RetrievalExample(tokens, labels, tuple(answers), {
            "task": "mqar", "sequence_length": length, "key_count": key_count,
            "query_distance": realized_distance, "requested_query_distance": distance,
            "query_distance_definition": "filler tokens between final memory token and first query marker",
            "memory_load": key_count,
        })

    def _overwrite(
        self, rng: random.Random, length: int, key_count: int, distance: int, overwrite_count: int
    ) -> RetrievalExample:
        keys, values = self._symbols(rng, key_count)
        target_key, initial = keys[0], values[0]
        stale_values = [initial]
        memory = [symbol for pair in zip(keys, values) for symbol in pair]
        latest = initial
        value_pool = list(range(self.config.vocabulary_size // 2, self.config.vocabulary_size))
        for _ in range(overwrite_count):
            candidate = rng.choice(value_pool)
            while candidate == latest:
                candidate = rng.choice(value_pool)
            stale_values.append(latest)
            latest = candidate
            memory.extend((target_key, latest))
        stale_values = stale_values[:-1]
        unrelated = 1 if key_count > 1 else None
        groups = [[QUERY_MARKER, target_key, latest]]
        if unrelated is not None:
            groups.append([QUERY_MARKER, keys[unrelated], values[unrelated]])
        required = len(memory) + sum(len(group) for group in groups)
        realized_distance = min(max(0, distance), max(0, length - required))
        prefix_filler = [FILLER_TOKEN] * max(0, length - required - realized_distance)
        gap_filler = [FILLER_TOKEN] * realized_distance
        sequence = prefix_filler + memory + gap_filler + [symbol for group in groups for symbol in group]
        sequence.extend([FILLER_TOKEN] * (length - len(sequence)))
        answers: list[AnswerPosition] = []
        cursor = len(prefix_filler) + len(memory) + realized_distance
        answers.append(AnswerPosition(cursor + 1, latest, tuple(stale_values), "latest", overwrite_count, realized_distance))
        if unrelated is not None and cursor + 4 < len(sequence):
            answers.append(AnswerPosition(cursor + 4, values[unrelated], (), "unrelated", 0, realized_distance))
        tokens, labels = self._labels(sequence, answers)
        return RetrievalExample(tokens, labels, tuple(answers), {
            "task": "overwrite", "sequence_length": length, "key_count": key_count,
            "query_distance": realized_distance, "requested_query_distance": distance,
            "query_distance_definition": "filler tokens between final memory token and first query marker",
            "memory_load": key_count + overwrite_count,
            "overwrite_count": overwrite_count,
        })

    def manifest(self) -> dict[str, Any]:
        data = {
            "generator_version": GENERATOR_VERSION, "task": self.task, "examples": self.examples,
            "seed": self.seed, "split": self.split, "sequence_length": self.sequence_length,
            "parameters": asdict(self.config),
        }
        data["hash"] = sha256(canonical_json(data).encode()).hexdigest()
        return data


def retrieval_dataloader(dataset: RetrievalDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate_retrieval(model: GatedDeltaNetLM, dataset: RetrievalDataset, device: torch.device) -> dict[str, Any]:
    model_was_training = model.training
    model.eval()
    total = correct = latest_total = latest_correct = stale_errors = unrelated_total = unrelated_correct = 0
    breakdown: dict[str, dict[str, list[int]]] = {
        key: {} for key in (
            "sequence_length", "key_count", "requested_query_distance", "query_distance",
            "memory_load", "overwrite_count",
        )
    }
    for index in range(len(dataset)):
        example = dataset.example(index)
        logits, _ = model(example.tokens.unsqueeze(0).to(device))
        predictions = logits[0].argmax(dim=-1).cpu()
        for answer in example.answers:
            predicted = int(predictions[answer.label_position])
            is_correct = predicted == answer.target
            total += 1
            correct += is_correct
            if answer.kind == "latest":
                latest_total += 1
                latest_correct += is_correct
                stale_errors += predicted in answer.stale_values
            elif answer.kind == "unrelated":
                unrelated_total += 1
                unrelated_correct += is_correct
            for key, value in example.metadata.items():
                if key in breakdown:
                    bucket = breakdown[key].setdefault(str(value), [0, 0])
                    bucket[0] += int(is_correct)
                    bucket[1] += 1
    model.train(model_was_training)
    formatted = {key: {group: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]} for group, values in groups.items()} for key, groups in breakdown.items()}
    result: dict[str, Any] = {
        "generator_manifest": dataset.manifest(), "exact_query_accuracy": correct / total if total else None,
        "correct": correct, "total": total, "breakdowns": formatted,
    }
    if dataset.task == "overwrite":
        result.update({
            "latest_value_accuracy": latest_correct / latest_total if latest_total else None,
            "stale_value_error_rate": stale_errors / latest_total if latest_total else None,
            "unrelated_association_retention": unrelated_correct / unrelated_total if unrelated_total else None,
        })
    return result
