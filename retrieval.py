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
# Shared corrected identity: both synthetic families use the same protocol
# version so a corrected study cannot accidentally pair a repaired overwrite
# generator with the legacy MQAR semantics.
CORRECTED_OVERWRITE_GENERATOR_VERSION = "mqar-overwrite-v4"
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
        if self.generator_version not in {
            GENERATOR_VERSION, AUDITED_GENERATOR_VERSION, CORRECTED_OVERWRITE_GENERATOR_VERSION,
        }:
            raise ValueError(f"unsupported retrieval generator version {self.generator_version!r}")

    def __len__(self) -> int:
        return self.examples

    def _rng(self, index: int) -> random.Random:
        domain = {"train": 1, "held_out": 2, "validation": 3, "test": 4}[self.split]
        return random.Random((self.seed << 24) ^ (domain << 20) ^ index)

    def _difficulty(self, index: int) -> tuple[int, int, int, int]:
        if self.generator_version in {AUDITED_GENERATOR_VERSION, CORRECTED_OVERWRITE_GENERATOR_VERSION}:
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
        if self.generator_version in {AUDITED_GENERATOR_VERSION, CORRECTED_OVERWRITE_GENERATOR_VERSION}:
            if length <= 0 or key_count <= 0 or distance < 1 or overwrite_count < 0:
                raise ValueError("audited retrieval difficulty values are outside their valid ranges")
            if self.task == "mqar":
                if self.generator_version == CORRECTED_OVERWRITE_GENERATOR_VERSION:
                    return self._mqar_v4(rng, length, key_count, distance)
                return self._mqar_v3(rng, length, key_count, distance)
            if self.generator_version == CORRECTED_OVERWRITE_GENERATOR_VERSION:
                return self._overwrite_v4(rng, length, key_count, distance, overwrite_count)
            return self._overwrite_v3(rng, length, key_count, distance, overwrite_count)
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

    def _overwrite_v4(
        self, rng: random.Random, length: int, key_count: int, distance: int, overwrite_count: int,
    ) -> RetrievalExample:
        """Generate keyed overwrite cases without fixed-position solutions.

        The target's latest write is placed at a random position in an
        interleaved write stream, with at least one unrelated write after it.
        The target and unrelated queries are randomly ordered whenever the
        requested distance leaves enough causal prefix for either order.  The
        complete sequence is used: there is no suffix padding after the last
        scored answer.
        """
        if key_count < 2:
            raise ValueError("infeasible overwrite combination: keyed v4 requires at least two keys")
        if overwrite_count < 1:
            raise ValueError("infeasible overwrite combination: keyed v4 requires at least one rewrite")
        if distance < 1:
            raise ValueError("infeasible overwrite combination: keyed v4 query distance must be positive")
        if key_count > (self.config.vocabulary_size // 2 - 2):
            raise ValueError(f"infeasible overwrite key_count={key_count} for vocabulary_size={self.config.vocabulary_size}")

        keys, values = self._symbols(rng, key_count)
        target_index = rng.randrange(key_count)
        target_key = keys[target_index]
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
        rewrites = rng.sample(value_pool, overwrite_count)
        target_writes = [(target_key, values[target_index], "target-initial")]
        target_writes.extend((target_key, value, "target-rewrite") for value in rewrites)
        stale_values = tuple(value for _, value, _ in target_writes[:-1])
        unrelated_writes = [
            (keys[index], values[index], "unrelated")
            for index in range(key_count) if index != target_index
        ]
        latest_write = target_writes[-1]
        prior_target_writes = list(target_writes[:-1])
        write_count = len(prior_target_writes) + len(unrelated_writes) + 1
        # The latest target write must be after every prior target write, before
        # at least one unrelated write, and far enough back that the requested
        # query distance reaches the end of the memory prefix.  Choose
        # uniformly from those feasible interior positions instead of
        # shuffling and rejecting otherwise valid examples.
        minimum_index = max(
            len(prior_target_writes),
            (2 * write_count - distance - 1 + 1) // 2,
        )
        maximum_index = min(
            write_count - 2,
            (length - distance - 7) // 2,
        )
        if minimum_index > maximum_index:
            raise ValueError(
                f"infeasible overwrite combination: length={length}, key_count={key_count}, "
                f"query_distance={distance}, overwrite_count={overwrite_count}; "
                "no feasible interior latest-write position"
            )
        latest_index = rng.randint(minimum_index, maximum_index)
        rng.shuffle(unrelated_writes)
        prefix_unrelated_count = latest_index - len(prior_target_writes)
        prefix_unrelated = unrelated_writes[:prefix_unrelated_count]
        suffix_unrelated = unrelated_writes[prefix_unrelated_count:]
        prefix_slots: list[tuple[int, int, str] | None] = [None] * latest_index
        for slot, write in zip(sorted(rng.sample(range(latest_index), len(prior_target_writes))), prior_target_writes):
            prefix_slots[slot] = write
        unrelated_cursor = iter(prefix_unrelated)
        prefix = [next(unrelated_cursor) if slot is None else slot for slot in prefix_slots]
        writes = [*prefix, latest_write, *suffix_unrelated]
        post_target_writes = writes[latest_index + 1:]
        if not post_target_writes or any(write[0] == target_key for write in post_target_writes):
            raise AssertionError("keyed overwrite latest target write must be followed only by unrelated writes")

        memory = [token for key, value, _ in writes for token in (key, value)]
        latest_value_base = 2 * latest_index + 1
        memory_length = len(memory)
        if latest_value_base >= memory_length - 2:
            raise AssertionError("keyed overwrite must retain an unrelated write after the target")

        # A target-first query has an unrelated query after it; target-second
        # has the unrelated query immediately before it.  Keep only orders
        # whose markers remain after the complete memory prefix.
        order_options: list[tuple[bool, int]] = []
        for target_first, tail_tokens in ((True, 6), (False, 3)):
            target_query_base = latest_value_base + distance
            unrelated_query_base = target_query_base + (3 if target_first else -3)
            if target_query_base < memory_length or unrelated_query_base < memory_length:
                continue
            prefix_tokens = length - (target_query_base + tail_tokens)
            if prefix_tokens >= 0:
                order_options.append((target_first, prefix_tokens))
        if not order_options:
            raise ValueError(
                f"infeasible overwrite combination: length={length}, key_count={key_count}, "
                f"query_distance={distance}, overwrite_count={overwrite_count}; "
                "requested distance cannot fit randomized keyed queries after the memory prefix"
            )
        target_first, prefix_tokens = rng.choice(order_options)
        target_query_position = prefix_tokens + latest_value_base + distance
        unrelated_index = next(index for index in range(key_count) if index != target_index)
        unrelated_key = keys[unrelated_index]
        unrelated_value_position = next(
            2 * index + 1 for index, (key, _, _) in enumerate(writes)
            if key == unrelated_key
        ) + prefix_tokens
        unrelated_query_position = target_query_position + (3 if target_first else -3)
        prefix = [FILLER_TOKEN] * prefix_tokens
        first_query_position = target_query_position if target_first else unrelated_query_position
        gap = [FILLER_TOKEN] * (first_query_position - (prefix_tokens + memory_length))
        target_query = [QUERY_MARKER, target_key, target_writes[-1][1]]
        unrelated_query = [QUERY_MARKER, unrelated_key, values[unrelated_index]]
        queries = target_query + unrelated_query if target_first else unrelated_query + target_query
        sequence = prefix + memory + gap + queries
        if len(sequence) != length:
            raise AssertionError(f"keyed overwrite generator produced {len(sequence)} tokens for requested length {length}")
        oracle_target = self._audit_oracle(
            sequence, target_key, prefix_tokens + latest_value_base,
            target_query_position, self.config.vocabulary_size,
        )
        if oracle_target != target_writes[-1][1]:
            raise AssertionError("keyed overwrite oracle disagrees with latest target write")
        oracle_unrelated = self._audit_oracle(
            sequence, unrelated_key, unrelated_value_position,
            unrelated_query_position, self.config.vocabulary_size,
        )
        if oracle_unrelated != values[unrelated_index]:
            raise AssertionError("keyed overwrite oracle disagrees with unrelated write")
        target_answer = AnswerPosition(
            target_query_position + 1, oracle_target, stale_values, "latest", overwrite_count,
            distance, target_query_position, prefix_tokens + latest_value_base,
        )
        unrelated_answer = AnswerPosition(
            unrelated_query_position + 1, oracle_unrelated, (), "unrelated", 0,
            unrelated_query_position - unrelated_value_position,
            unrelated_query_position, unrelated_value_position,
        )
        answers = (target_answer, unrelated_answer)
        tokens, labels = self._labels(sequence, list(answers))
        target_write_positions = [
            prefix_tokens + 2 * index + 1 for index, (key, _, _) in enumerate(writes) if key == target_key
        ]
        return RetrievalExample(tokens, labels, answers, {
            "task": "overwrite", "sequence_length": length, "key_count": key_count,
            "query_count": len(answers), "query_distance": distance,
            "requested_query_distance": distance,
            "query_distance_definition": "target query marker minus target latest value position",
            "memory_load": len(writes), "overwrite_count": overwrite_count,
            "target_key": target_key, "target_first": target_first,
            "query_order": ["target", "unrelated"] if target_first else ["unrelated", "target"],
            "query_positions": [answer.query_position for answer in answers],
            "latest_value_positions": [answer.latest_value_position for answer in answers],
            "answer_distances": [answer.distance for answer in answers],
            "write_positions": target_write_positions,
            "all_write_positions": [prefix_tokens + 2 * index + 1 for index in range(len(writes))],
            "write_values": [value for key, value, _ in writes if key == target_key],
            "write_key_positions": [prefix_tokens + 2 * index for index, (key, _, _) in enumerate(writes) if key == target_key],
            "post_target_write_count": len(writes) - latest_index - 1,
            "prefix_filler_tokens": prefix_tokens,
            "suffix_padding_tokens": 0,
            "effective_sequence_tokens": len(sequence),
            "oracle": "causal-last-write-wins-v4",
        })

    def _mqar_v4(self, rng: random.Random, length: int, key_count: int, distance: int) -> RetrievalExample:
        """Generate MQAR with independent memory and query permutations.

        The complete causal prefix is constructed before labels are derived.
        Increasing the requested sequence length adds unqueried key/value
        writes before the queries, so a length sweep increases meaningful
        memory load rather than appending irrelevant suffix tokens.
        """
        if key_count > (self.config.vocabulary_size // 2 - 2):
            raise ValueError(f"infeasible mqar key_count={key_count} for vocabulary_size={self.config.vocabulary_size}")
        if key_count < 1 or distance < 1:
            raise ValueError("infeasible mqar combination: key_count and query distance must be positive")

        # Five tokens per queried association/query (two memory tokens and
        # three query tokens), plus the requested end-gap, are irreducible.
        minimum_length = 5 * key_count + distance
        if length < minimum_length:
            raise ValueError(
                f"infeasible mqar combination: length={length}, key_count={key_count}, "
                f"query_distance={distance}; minimum length is {minimum_length}"
            )

        keys, values = self._symbols(rng, key_count)
        memory_order = list(range(key_count))
        query_order = list(range(key_count))
        rng.shuffle(memory_order)
        rng.shuffle(query_order)

        extra_tokens = length - minimum_length
        if extra_tokens % 2:
            raise ValueError(
                f"infeasible mqar combination: length={length} leaves an odd number of "
                "tokens for key/value distractor writes"
            )
        distractor_count = extra_tokens // 2
        key_upper = self.config.vocabulary_size // 2
        distractor_keys = [key for key in range(2, key_upper) if key not in keys]
        if not distractor_keys and distractor_count:
            raise ValueError("infeasible mqar combination: no unqueried distractor key is available")
        distractor_writes = [
            (rng.choice(distractor_keys), rng.randrange(key_upper, self.config.vocabulary_size))
            for _ in range(distractor_count)
        ]
        writes = [(keys[index], values[index]) for index in memory_order]
        writes.extend(distractor_writes)
        rng.shuffle(writes)
        memory = [symbol for key, value in writes for symbol in (key, value)]

        memory_start = 0
        latest_positions = {
            key: memory_start + 2 * position + 1
            for position, (key, _) in enumerate(writes)
            if key in keys
        }
        first_query_position = memory_start + len(memory) + distance
        sequence = memory
        sequence += [FILLER_TOKEN] * distance

        answers: list[AnswerPosition] = []
        query_positions: list[int] = []
        value_positions: list[int] = []
        for item_index, item in enumerate(query_order):
            query_position = first_query_position + 3 * item_index
            sequence.extend((QUERY_MARKER, keys[item], values[item]))
            value_position = latest_positions[keys[item]]
            oracle_target = self._audit_oracle(
                sequence, keys[item], value_position, query_position, self.config.vocabulary_size,
            )
            if oracle_target != values[item]:
                raise AssertionError("MQAR lookup oracle disagrees with generated answer")
            query_positions.append(query_position)
            value_positions.append(value_position)
            answers.append(AnswerPosition(
                query_position + 1, oracle_target, (), "mqar", 0,
                query_position - value_position, query_position, value_position,
            ))
        if len(sequence) != length:
            raise AssertionError(f"corrected MQAR generator produced {len(sequence)} tokens for requested length {length}")

        tokens, labels = self._labels(sequence, answers)
        return RetrievalExample(tokens, labels, tuple(answers), {
            "task": "mqar", "sequence_length": length, "key_count": key_count,
            "query_count": key_count, "query_distance": distance,
            "requested_query_distance": distance,
            "query_distance_definition": "query marker minus that key's latest value position",
            "memory_order": memory_order, "query_order": query_order,
            "query_positions": query_positions, "latest_value_positions": value_positions,
            "answer_distances": [answer.distance for answer in answers],
            "memory_load": len(writes), "distractor_count": distractor_count,
            "distractor_write_count": distractor_count,
            "distractor_key_count": len(set(key for key, _ in distractor_writes)),
            "end_gap_tokens": distance, "prefix_filler_tokens": 0,
            "suffix_padding_tokens": 0, "effective_sequence_tokens": len(tokens),
            "oracle": "causal-last-write-wins-v4",
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
    history_conditioning: Literal["answer_free", "teacher_forced"] = "answer_free",
) -> dict[str, Any]:
    if history_conditioning not in {"answer_free", "teacher_forced"}:
        raise ValueError("history_conditioning must be 'answer_free' or 'teacher_forced'")
    model_was_training = model.training
    model.eval()
    indices = tuple(range(len(dataset))) if example_indices is None else tuple(example_indices)
    if not indices:
        raise ValueError("Retrieval evaluation requires at least one example index")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Retrieval evaluation indices must be unique and within the dataset")
    total = correct = latest_total = latest_correct = stale_errors = other_latest_errors = unrelated_total = unrelated_correct = 0
    complete_examples = 0
    first_query_total = first_query_correct = later_query_total = later_query_correct = 0
    supervised_tokens = effective_tokens = 0
    joint_key_distance: dict[str, dict[str, list[int]]] = {}
    breakdown: dict[str, dict[str, list[int]]] = {
        key: {} for key in (
            "sequence_length", "key_count", "requested_query_distance", "query_distance",
            "memory_load", "overwrite_count",
        )
    }
    for index in indices:
        example = dataset.example(index)
        supervised_tokens += int((example.labels != -100).sum().item())
        effective_tokens += int(example.tokens.numel())
        model_tokens = example.tokens.clone()
        if history_conditioning == "answer_free":
            # The label is scored at the key position; replace the answer
            # token one position later so later queries cannot receive an
            # earlier gold answer through the teacher-forced history.
            for answer in example.answers:
                answer_token_position = answer.label_position + 1
                if answer_token_position < model_tokens.numel():
                    model_tokens[answer_token_position] = FILLER_TOKEN
        logits, _ = model(model_tokens.unsqueeze(0).to(device))
        predictions = logits[0].argmax(dim=-1).cpu()
        example_correct = 0
        ordered_answers = tuple(sorted(example.answers, key=lambda answer: answer.label_position))
        for ordinal, answer in enumerate(ordered_answers):
            predicted = int(predictions[answer.label_position])
            is_correct = predicted == answer.target
            total += 1
            correct += is_correct
            example_correct += int(is_correct)
            if ordinal == 0:
                first_query_total += 1
                first_query_correct += int(is_correct)
            else:
                later_query_total += 1
                later_query_correct += int(is_correct)
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
        complete_examples += int(example_correct == len(ordered_answers))
    model.train(model_was_training)
    formatted = {key: {group: {"correct": values[0], "total": values[1], "accuracy": values[0] / values[1]} for group, values in groups.items()} for key, groups in breakdown.items()}
    result: dict[str, Any] = {
        "generator_manifest": dataset.manifest(), "exact_query_accuracy": correct / total if total else None,
        "query_accuracy": correct / total if total else None,
        "example_accuracy": complete_examples / len(indices),
        "all_queries_correct": complete_examples,
        "example_total": len(indices),
        "macro_example_accuracy": complete_examples / len(indices),
        "supervised_answer_tokens": supervised_tokens,
        "effective_input_tokens": effective_tokens,
        "first_query_accuracy": first_query_correct / first_query_total if first_query_total else None,
        "later_query_accuracy": later_query_correct / later_query_total if later_query_total else None,
        "history_conditioning": history_conditioning,
        "metric_contract": {
            "primary_weighting": "micro_query",
            "query_accuracy_denominator": "all_scored_queries",
            "example_accuracy_denominator": "examples_with_all_scored_queries_correct",
            "difficulty_distance": "actual_answer_query_marker_minus_latest_value_position",
            "requested_distance_field": "generator_manifest.parameters.query_distances",
            "history_conditioning": history_conditioning,
            "query_format": "query_marker_key_then_answer_token_label",
            "answer_free_primary": history_conditioning == "answer_free",
            "causal_prediction": "logit_at_key_position_predicts_following_answer",
        },
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


def evaluate_retrieval_baselines(
    dataset: RetrievalDataset,
    *,
    example_indices: Sequence[int] | None = None,
    frequency_dataset: RetrievalDataset | None = None,
    shortcut_max_accuracy: float = 0.90,
) -> dict[str, Any]:
    """Evaluate key-free retrieval baselines without using placement metadata.

    The evaluator reconstructs each causal memory prefix from emitted tokens.
    It never uses ``latest_value_position`` or any generator placement field.
    ``frequency_dataset`` must be a training split when supplied; validation
    and held-out baselines therefore cannot fit their no-memory prediction on
    the evaluated cases.
    """

    if not 0.0 <= shortcut_max_accuracy <= 1.0:
        raise ValueError("shortcut_max_accuracy must be between 0 and 1")
    if frequency_dataset is not None:
        if frequency_dataset.split != "train":
            raise ValueError("frequency_dataset must be a training split")
        if frequency_dataset.task != dataset.task:
            raise ValueError("frequency_dataset must use the same retrieval task")
    indices = tuple(range(len(dataset))) if example_indices is None else tuple(example_indices)
    if not indices:
        raise ValueError("Baseline evaluation requires at least one example index")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Baseline evaluation indices must be unique and within the dataset")

    value_floor = dataset.config.vocabulary_size // 2
    frequency_counts: dict[int, int] = {}
    if frequency_dataset is not None:
        for index in range(len(frequency_dataset)):
            tokens = frequency_dataset.example(index).tokens.tolist()
            for token in tokens:
                if token >= value_floor:
                    frequency_counts[token] = frequency_counts.get(token, 0) + 1
    frequency_total = sum(frequency_counts.values())
    frequency_prediction = max(frequency_counts, key=frequency_counts.get) if frequency_counts else None
    frequency_chance = (
        frequency_counts[frequency_prediction] / frequency_total
        if frequency_prediction is not None and frequency_total else None
    )

    baseline_names = (
        "keyed_last_write_oracle", "first_value", "last_value", "most_recent_value",
        "final_two_values", "no_memory_frequency",
    )
    stats = {
        name: {"correct": 0, "total": 0, "accuracy": None}
        for name in baseline_names
    }
    by_cell: dict[str, dict[str, dict[str, int | float | None]]] = {}
    cell_checks: dict[str, dict[str, int]] = {}
    oracle_mismatches = 0
    oracle_checks = 0
    incomplete_checks = 0
    counterfactual_by_cell: dict[str, dict[str, int]] = {}
    counterfactual_checks = counterfactual_passes = counterfactual_mismatches = counterfactual_incomplete = 0

    def latest_value_for_key(tokens: list[int], key: int, stop: int) -> int | None:
        writes = [
            tokens[position + 1]
            for position in range(stop)
            if tokens[position] == key
            and position + 1 < stop
            and tokens[position + 1] >= value_floor
        ]
        return writes[-1] if writes else None

    def record(name: str, prediction: int | None, target: int, cell: str) -> None:
        checks = cell_checks.setdefault(cell, {"expected": 0, "incomplete": 0})
        checks["expected"] += 1
        if prediction is None:
            nonlocal incomplete_checks
            incomplete_checks += 1
            checks["incomplete"] += 1
            return
        entry = stats[name]
        entry["total"] += 1
        entry["correct"] += int(prediction == target)
        cell_entry = by_cell.setdefault(cell, {}).setdefault(name, {"correct": 0, "total": 0})
        cell_entry["total"] += 1
        cell_entry["correct"] += int(prediction == target)

    for index in indices:
        example = dataset.example(index)
        tokens = example.tokens.tolist()
        markers = [position for position, token in enumerate(tokens) if token == QUERY_MARKER]
        if not markers:
            raise ValueError(f"example {index} has no query marker")
        first_query_marker = markers[0]
        memory_prefix = tokens[:first_query_marker]
        memory_values = [token for token in memory_prefix if token >= value_floor]
        if not memory_values:
            raise ValueError(f"example {index} has no value tokens in its causal memory prefix")
        distinct_values = max(1, len(set(memory_values)))
        answer_order = tuple(sorted(example.answers, key=lambda answer: answer.label_position))
        for ordinal, answer in enumerate(answer_order):
            query_marker = answer.label_position - 1
            if query_marker < 0 or query_marker >= len(tokens) or tokens[query_marker] != QUERY_MARKER:
                raise ValueError(f"example {index} has an answer without a causal query marker")
            query_key = tokens[answer.label_position]
            writes = [
                tokens[position + 1]
                for position in range(first_query_marker)
                if tokens[position] == query_key
                and position + 1 < first_query_marker
                and tokens[position + 1] >= value_floor
            ]
            oracle_target = writes[-1] if writes else None
            target = int(example.labels[answer.label_position])
            oracle_checks += 1
            if oracle_target != target:
                oracle_mismatches += 1
            cell = canonical_json({
                "sequence_length": example.metadata.get("sequence_length"),
                "key_count": example.metadata.get("key_count"),
                "requested_query_distance": example.metadata.get("requested_query_distance"),
                "overwrite_count": answer.overwrite_count,
            })
            record("keyed_last_write_oracle", oracle_target, target, cell)
            record("first_value", memory_values[0], target, cell)
            record("last_value", memory_values[-1], target, cell)
            record("most_recent_value", memory_values[-1], target, cell)
            final_two_prediction = memory_values[-2 + ordinal] if 0 <= ordinal < 2 and len(memory_values) >= 2 else None
            record("final_two_values", final_two_prediction, target, cell)
            record("no_memory_frequency", frequency_prediction, target, cell)

            # Counterfactual query-key audit: keep the entire causal memory
            # prefix fixed, change only the queried key, and derive the new
            # target by an independent token-prefix scan.
            key_values = {
                key: latest_value_for_key(tokens, key, first_query_marker)
                for key in set(memory_prefix)
                if 2 <= key < value_floor
            }
            alternatives = sorted(
                key for key, value in key_values.items()
                if key != query_key and value is not None and value != oracle_target
            )
            counterfactual_checks += 1
            cell_counterfactual = counterfactual_by_cell.setdefault(
                cell, {"checks": 0, "passes": 0, "mismatches": 0, "incomplete": 0}
            )
            cell_counterfactual["checks"] += 1
            if not alternatives or oracle_target is None:
                counterfactual_incomplete += 1
                cell_counterfactual["incomplete"] += 1
            else:
                counterfactual_tokens = list(tokens)
                counterfactual_tokens[answer.label_position] = alternatives[0]
                counterfactual_target = latest_value_for_key(
                    counterfactual_tokens, alternatives[0], first_query_marker
                )
                if (
                    counterfactual_tokens[answer.label_position] != query_key
                    and counterfactual_target is not None
                    and counterfactual_target != oracle_target
                    and counterfactual_target == key_values[alternatives[0]]
                ):
                    counterfactual_passes += 1
                    cell_counterfactual["passes"] += 1
                else:
                    counterfactual_mismatches += 1
                    cell_counterfactual["mismatches"] += 1

    for entry in stats.values():
        if entry["total"]:
            entry["accuracy"] = entry["correct"] / entry["total"]
    shortcut_names = tuple(name for name in baseline_names if name != "keyed_last_write_oracle")
    shortcut_cells: dict[str, dict[str, Any]] = {}
    for cell, cells in by_cell.items():
        for entry in cells.values():
            entry["accuracy"] = entry["correct"] / entry["total"] if entry["total"] else None
        available = [cells[name]["accuracy"] for name in shortcut_names if name in cells and cells[name]["accuracy"] is not None]
        maximum = max(available) if available else None
        checks = cell_checks.get(cell, {"expected": 0, "incomplete": 0})
        shortcut_cells[cell] = {
            "max_key_free_accuracy": maximum,
            "threshold": shortcut_max_accuracy,
            "pass": maximum is not None and maximum <= shortcut_max_accuracy and checks["incomplete"] == 0,
            "incomplete_checks": checks["incomplete"],
        }
    for cell, checks in cell_checks.items():
        shortcut_cells.setdefault(cell, {
            "max_key_free_accuracy": None,
            "threshold": shortcut_max_accuracy,
            "pass": False,
            "incomplete_checks": checks["incomplete"],
        })
    oracle_pass = oracle_mismatches == 0 and oracle_checks > 0
    all_shortcuts_pass = bool(shortcut_cells) and all(item["pass"] for item in shortcut_cells.values())
    counterfactual_pass = counterfactual_checks > 0 and counterfactual_incomplete == 0 and counterfactual_mismatches == 0
    if not oracle_pass or incomplete_checks or not counterfactual_pass:
        status = "failed"
    elif not all_shortcuts_pass:
        status = "shortcut_detected"
    else:
        status = "complete"
    chance = {
        "keyed_last_write_oracle": 1.0,
        "first_value": "1 / distinct_memory_values",
        "last_value": "1 / distinct_memory_values",
        "most_recent_value": "1 / distinct_memory_values",
        "final_two_values": "min(2 / distinct_memory_values, 1)",
        "no_memory_frequency": frequency_chance,
    }
    return {
        "status": status,
        "task": dataset.task,
        "generator_manifest": dataset.manifest(),
        "evaluated_examples": len(indices),
        "example_indices_sha256": sha256(",".join(map(str, indices)).encode()).hexdigest(),
        "baselines": stats,
        "chance_expectation": chance,
        "frequency_fit": {
            "source_split": frequency_dataset.split if frequency_dataset is not None else None,
            "source_examples": len(frequency_dataset) if frequency_dataset is not None else 0,
            "value_count": len(frequency_counts),
            "observed_value_tokens": frequency_total,
            "prediction": frequency_prediction,
        },
        "oracle_audit": {
            "independent_last_write_oracle": "causal token-prefix scan",
            "target_mismatches": oracle_mismatches,
            "target_checks": oracle_checks,
            "pass": oracle_pass,
        },
        "counterfactual_query_key_audit": {
            "method": "replace query key while preserving the causal memory prefix",
            "target_source": "independent last-write token-prefix scan",
            "checks": counterfactual_checks,
            "passes": counterfactual_passes,
            "mismatches": counterfactual_mismatches,
            "incomplete": counterfactual_incomplete,
            "pass": counterfactual_pass,
            "by_difficulty_cell": counterfactual_by_cell,
        },
        "by_difficulty_cell": by_cell,
        "shortcut_criterion": {
            "max_accuracy": shortcut_max_accuracy,
            "baseline_names": list(shortcut_names),
            "by_cell": shortcut_cells,
            "all_cells_pass": all_shortcuts_pass,
            "scope": "shortcut_audit_only; does not qualify a trained model or research protocol",
        },
        "completeness": {
            "incomplete_checks": incomplete_checks,
            "oracle_checks": oracle_checks,
            "oracle_mismatches": oracle_mismatches,
        },
    }
