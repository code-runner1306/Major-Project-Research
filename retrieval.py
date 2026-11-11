"""Versioned deterministic MQAR and overwrite-recall data and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import random
from typing import Any, Literal, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from artifacts import canonical_json
from config import RetrievalConfig
from model import GatedDeltaNetLM


GENERATOR_VERSION = "mqar-overwrite-v2"
AUDITED_GENERATOR_VERSION = "mqar-overwrite-v3"
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
    query_position: int | None = None
    latest_value_position: int | None = None


@dataclass(frozen=True)
class RetrievalExample:
    tokens: torch.Tensor
    labels: torch.Tensor
    answers: tuple[AnswerPosition, ...]
    metadata: dict[str, Any]


class RetrievalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Indexed, generated-on-demand data so the 100k production set stays compact."""

    def __init__(
        self, *, task: Literal["mqar", "overwrite"], examples: int, config: RetrievalConfig,
        seed: int, split: Literal["train", "validation", "held_out", "test"], sequence_length: int | None = None,
    ):
        if config.vocabulary_size < 16:
            raise ValueError("Retrieval vocabulary must reserve enough symbolic keys and values")
        self.task, self.examples, self.config, self.seed, self.split = task, examples, config, seed, split
        self.sequence_length = sequence_length
        self.generator_version = getattr(config, "generator_version", GENERATOR_VERSION)
        if self.generator_version not in {GENERATOR_VERSION, AUDITED_GENERATOR_VERSION}:
            raise ValueError(f"unsupported retrieval generator version {self.generator_version!r}")

    def __len__(self) -> int:
        return self.examples

    def _rng(self, index: int) -> random.Random:
        domain = {"train": 1, "held_out": 2, "validation": 3, "test": 4}[self.split]
        return random.Random((self.seed << 24) ^ (domain << 20) ^ index)

    def _difficulty(self, index: int) -> tuple[int, int, int, int]:
        if self.generator_version == AUDITED_GENERATOR_VERSION:
            lengths, keys, distances, overwrites = (
                self.config.sequence_lengths, self.config.key_counts,
                self.config.query_distances, self.config.overwrite_counts,
            )
            if not all((lengths, keys, distances, overwrites)):
                raise ValueError("audited retrieval difficulty grids must be non-empty")
            difficulty_count = len(lengths) * len(keys) * len(distances) * len(overwrites)
            slot = index % difficulty_count
            length_i = slot % len(lengths)
            slot //= len(lengths)
            key_i = slot % len(keys)
            slot //= len(keys)
            distance_i = slot % len(distances)
            slot //= len(distances)
            overwrite_i = slot % len(overwrites)
            return (self.sequence_length or lengths[length_i], keys[key_i], distances[distance_i], overwrites[overwrite_i])
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
        if self.generator_version == AUDITED_GENERATOR_VERSION:
            if length <= 0 or key_count <= 0 or distance < 1 or overwrite_count < 0:
                raise ValueError("audited retrieval difficulty values are outside their valid ranges")
            return self._mqar_v3(rng, length, key_count, distance) if self.task == "mqar" else self._overwrite_v3(
                rng, length, key_count, distance, overwrite_count
            )
        if self.task == "mqar":
            return self._mqar(rng, length, key_count, distance)
        return self._overwrite(rng, length, key_count, distance, overwrite_count)

    @staticmethod
    def _audit_oracle(
        sequence: list[int], target_key: int, latest_value_position: int,
        query_position: int, vocabulary_size: int,
    ) -> int:
        if not (0 <= latest_value_position < query_position < len(sequence)):
            raise ValueError("audited example positions fall outside the generated sequence")
        if sequence[latest_value_position - 1] != target_key:
            raise AssertionError("oracle target value is not preceded by the queried key")
        if sequence[query_position] != QUERY_MARKER or sequence[query_position + 1] != target_key:
            raise AssertionError("oracle query does not reference the intended key")
        # Independent last-write-wins scan over the key/value stream before
        # query marker. Filler/query symbols cannot be values in this range.
        writes = [
            (position, sequence[position + 1])
            for position in range(query_position)
            if sequence[position] == target_key
            and sequence[position + 1] >= vocabulary_size // 2
        ]
        if not writes or writes[-1] != (latest_value_position - 1, sequence[latest_value_position]):
            raise AssertionError("last-write-wins oracle disagrees with latest relevant value position")
        return writes[-1][1]

    def _check_feasible(self, *, length: int, key_count: int, distance: int, overwrite_count: int, task: str) -> None:
        if key_count > (self.config.vocabulary_size // 2 - 2):
            raise ValueError(f"infeasible {task} key_count={key_count} for vocabulary_size={self.config.vocabulary_size}")
        writes = key_count + (overwrite_count if task == "overwrite" else 0)
        if task == "mqar":
            # All keys are queried. The requested distance anchors the first
            # answer after a 2*key_count memory block.
            minimum_length = 5 * key_count + distance - 1
        elif key_count > 1 and distance >= 3:
            # A distractor key/value write follows the latest target write.
            minimum_length = 2 * writes + distance + 3
        else:
            query_tokens = 6 if key_count > 1 else 3
            minimum_length = 2 * writes + distance + query_tokens - 1
        if length < minimum_length:
            raise ValueError(
                f"infeasible {task} combination: length={length}, key_count={key_count}, "
                f"query_distance={distance}, overwrite_count={overwrite_count}; minimum length is {minimum_length}"
            )

    def _mqar_v3(self, rng: random.Random, length: int, key_count: int, distance: int) -> RetrievalExample:
        self._check_feasible(length=length, key_count=key_count, distance=distance, overwrite_count=0, task="mqar")
        keys, values = self._symbols(rng, key_count)
        target_index = rng.randrange(key_count)
        order = list(range(key_count))
        rng.shuffle(order)
        order.remove(target_index)
        order.append(target_index)
        memory = [symbol for index in order for symbol in (keys[index], values[index])]
        latest_positions = {keys[item]: 2 * position + 1 for position, item in enumerate(order)}
        query_order = [target_index] + [item for item in range(key_count) if item != target_index]
        tail = query_order[1:]
        rng.shuffle(tail)
        query_order = [target_index, *tail]
        first_query_position = latest_positions[keys[target_index]] + distance
        sequence = list(memory)
        sequence += [FILLER_TOKEN] * (first_query_position - len(sequence))
        answers: list[AnswerPosition] = []
        query_positions: list[int] = []
        value_positions: list[int] = []
        for item_index, item in enumerate(query_order):
            query_position = first_query_position + 3 * item_index
            sequence.extend((QUERY_MARKER, keys[item], values[item]))
            oracle_target = self._audit_oracle(
                sequence, keys[item], latest_positions[keys[item]], query_position, self.config.vocabulary_size,
            )
            if oracle_target != values[item]:
                raise AssertionError("MQAR lookup oracle disagrees with generated answer")
            value_position = latest_positions[keys[item]]
            query_positions.append(query_position)
            value_positions.append(value_position)
            answers.append(AnswerPosition(
                query_position + 1, oracle_target, (), "mqar", 0,
                query_position - value_position, query_position, value_position,
            ))
        sequence += [FILLER_TOKEN] * (length - len(sequence))
        tokens, labels = self._labels(sequence, answers)
        return RetrievalExample(tokens, labels, tuple(answers), {
            "task": "mqar", "sequence_length": length, "key_count": key_count,
            "query_count": key_count,
            "query_distance": distance, "requested_query_distance": distance,
            "anchor_latest_value_position": latest_positions[keys[target_index]],
            "anchor_query_position": first_query_position,
            "query_positions": query_positions, "latest_value_positions": value_positions,
            "answer_distances": [answer.distance for answer in answers],
            "query_distance_definition": "requested distance anchors the first answer; each answer distance is its query marker position minus that key's latest relevant value position",
            "memory_load": key_count,
        })

    def _overwrite_v3(self, rng: random.Random, length: int, key_count: int, distance: int, overwrite_count: int) -> RetrievalExample:
        self._check_feasible(length=length, key_count=key_count, distance=distance, overwrite_count=overwrite_count, task="overwrite")
        keys, values = self._symbols(rng, key_count)
        target_index = rng.randrange(key_count)
        target_key = keys[target_index]
        target_writes = [(target_key, values[target_index], "initial")]
        stale_values: list[int] = []
        latest = values[target_index]
        initial_values = set(values)
        value_pool = [
            value for value in range(self.config.vocabulary_size // 2, self.config.vocabulary_size)
            if value not in initial_values
        ]
        if overwrite_count > len(value_pool):
            raise ValueError(
                f"infeasible overwrite combination: {overwrite_count} rewrites need distinct values, "
                f"but only {len(value_pool)} are available"
            )
        for candidate in rng.sample(value_pool, overwrite_count):
            target_writes.append((target_key, candidate, "overwrite"))
            stale_values.append(latest)
            latest = candidate
        distractors = [
            (keys[index], values[index], "unrelated")
            for index in range(key_count) if index != target_index
        ]
        rng.shuffle(distractors)
        post_target = distractors.pop() if distance >= 3 and distractors else None

        def interleave_prior_target_writes() -> list[tuple[int, int, str]]:
            result: list[tuple[int, int, str]] = []
            target_cursor = 0
            unrelated = list(distractors)
            while target_cursor < len(target_writes) - 1 or unrelated:
                if target_cursor < len(target_writes) - 1 and (not unrelated or rng.random() < 0.5):
                    result.append(target_writes[target_cursor])
                    target_cursor += 1
                else:
                    result.append(unrelated.pop())
            return result

        writes = interleave_prior_target_writes()
        writes.append(target_writes[-1])
        if post_target is not None:
            writes.append(post_target)
        memory = [token for key, value, _ in writes for token in (key, value)]
        target_write_indices = [index for index, (key, _, _) in enumerate(writes) if key == target_key]
        latest_value_position = 2 * target_write_indices[-1] + 1
        query_position = latest_value_position + distance
        sequence = list(memory)
        sequence += [FILLER_TOKEN] * (query_position - len(sequence))
        sequence += [QUERY_MARKER, target_key, latest]
        oracle_target = self._audit_oracle(sequence, target_key, latest_value_position, query_position, self.config.vocabulary_size)
        if oracle_target != latest:
            raise AssertionError("overwrite last-write-wins oracle disagrees with generated answer")
        answer = AnswerPosition(
            query_position + 1, oracle_target, tuple(stale_values), "latest", overwrite_count,
            distance, query_position, latest_value_position,
        )
        answers = [answer]
        query_positions = [query_position]
        latest_value_positions = [latest_value_position]
        unrelated_index = (
            keys.index(post_target[0]) if post_target is not None
            else next((index for index in range(key_count) if index != target_index), None)
        )
        if unrelated_index is not None:
            unrelated_key = keys[unrelated_index]
            unrelated_write_index = max(index for index, (key, _, _) in enumerate(writes) if key == unrelated_key)
            unrelated_value_position = unrelated_write_index * 2 + 1
            unrelated_query_position = query_position + 3
            oracle_unrelated = self._audit_oracle(
                sequence + [QUERY_MARKER, unrelated_key, values[unrelated_index]],
                unrelated_key, unrelated_value_position, unrelated_query_position, self.config.vocabulary_size,
            )
            if oracle_unrelated != values[unrelated_index]:
                raise AssertionError("overwrite unrelated-association oracle disagrees with generated answer")
            answers.append(AnswerPosition(
                unrelated_query_position + 1, oracle_unrelated, (), "unrelated", 0,
                unrelated_query_position - unrelated_value_position,
                unrelated_query_position, unrelated_value_position,
            ))
            query_positions.append(unrelated_query_position)
            latest_value_positions.append(unrelated_value_position)
            sequence.extend((QUERY_MARKER, unrelated_key, oracle_unrelated))
        sequence += [FILLER_TOKEN] * (length - len(sequence))
        tokens, labels = self._labels(sequence, answers)
        write_key_positions = [2 * index for index, (key, _, _) in enumerate(writes) if key == target_key]
        write_positions = [position + 1 for position in write_key_positions]
        write_values = [sequence[position] for position in write_positions]
        return RetrievalExample(tokens, labels, tuple(answers), {
            "task": "overwrite", "sequence_length": length, "key_count": key_count,
            "query_count": len(answers),
            "query_distance": distance, "requested_query_distance": distance,
            "latest_value_position": latest_value_position, "query_position": query_position,
            "query_positions": query_positions, "latest_value_positions": latest_value_positions,
            "answer_distances": [answer.distance for answer in answers],
            "query_distance_definition": "requested distance anchors the latest-value answer; each answer distance is its query marker position minus that key's latest relevant value position",
            "memory_load": key_count + overwrite_count, "overwrite_count": overwrite_count,
            "obsolete_write_count": overwrite_count, "obsolete_values": len(stale_values),
            "write_positions": write_positions,
            "write_values": write_values,
            "write_key_positions": write_key_positions,
            "obsolete_write_positions": write_positions[:-1],
        })

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
            "generator_version": self.generator_version, "task": self.task, "examples": self.examples,
            "seed": self.seed, "split": self.split, "sequence_length": self.sequence_length,
            "parameters": asdict(self.config),
        }
        data["hash"] = sha256(canonical_json(data).encode()).hexdigest()
        return data


def retrieval_dataloader(dataset: RetrievalDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate_retrieval(
    model: GatedDeltaNetLM, dataset: RetrievalDataset, device: torch.device,
    *, example_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    model_was_training = model.training
    model.eval()
    indices = tuple(range(len(dataset))) if example_indices is None else tuple(example_indices)
    if not indices:
        raise ValueError("Retrieval evaluation requires at least one example index")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Retrieval evaluation indices must be unique and within the dataset")
    total = correct = latest_total = latest_correct = stale_errors = other_latest_errors = unrelated_total = unrelated_correct = 0
    joint_key_distance: dict[str, dict[str, list[int]]] = {}
    breakdown: dict[str, dict[str, list[int]]] = {
        key: {} for key in (
            "sequence_length", "key_count", "requested_query_distance", "query_distance",
            "memory_load", "overwrite_count",
        )
    }
    for index in indices:
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
                if predicted in answer.stale_values:
                    stale_errors += 1
                elif not is_correct:
                    other_latest_errors += 1
            elif answer.kind == "unrelated":
                unrelated_total += 1
                unrelated_correct += is_correct
            for key, value in example.metadata.items():
                if key == "query_distance":
                    bucket = breakdown[key].setdefault(str(answer.distance), [0, 0])
                    bucket[0] += int(is_correct)
                    bucket[1] += 1
                elif key in breakdown:
                    bucket = breakdown[key].setdefault(str(value), [0, 0])
                    bucket[0] += int(is_correct)
                    bucket[1] += 1
            key_bucket = joint_key_distance.setdefault(str(example.metadata["key_count"]), {})
            joint = key_bucket.setdefault(str(answer.distance), [0, 0])
            joint[0] += int(is_correct)
            joint[1] += 1
    model.train(model_was_training)
    formatted = {key: {group: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]} for group, values in groups.items()} for key, groups in breakdown.items()}
    result: dict[str, Any] = {
        "generator_manifest": dataset.manifest(), "exact_query_accuracy": correct / total if total else None,
        "correct": correct, "total": total, "breakdowns": formatted,
        "evaluated_examples": len(indices),
        "example_indices_sha256": sha256(",".join(map(str, indices)).encode()).hexdigest(),
        "key_count_by_query_distance": {
            key_count: {
                distance: {"correct": counts[0], "total": counts[1], "accuracy": counts[0] / counts[1]}
                for distance, counts in sorted(distances.items(), key=lambda item: int(item[0]))
            }
            for key_count, distances in sorted(joint_key_distance.items(), key=lambda item: int(item[0]))
        },
    }
    if dataset.task == "overwrite":
        result.update({
            "latest_value_accuracy": latest_correct / latest_total if latest_total else None,
            "stale_value_error_rate": stale_errors / latest_total if latest_total else None,
            "stale_value_errors": stale_errors,
            "other_latest_value_errors": other_latest_errors,
            "other_latest_value_error_rate": other_latest_errors / latest_total if latest_total else None,
            "unrelated_association_retention": unrelated_correct / unrelated_total if unrelated_total else None,
        })
    return result
