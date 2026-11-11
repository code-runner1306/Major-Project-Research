from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest

from retrieval import AUDITED_GENERATOR_VERSION, RetrievalDataset


def _config(base, **overrides):
    values = asdict(base)
    values.update(overrides)
    values["generator_version"] = AUDITED_GENERATOR_VERSION
    return SimpleNamespace(**values)


@pytest.mark.parametrize("task", ["mqar", "overwrite"])
def test_audited_generator_records_actual_distance_and_oracle_trace(smoke_config, task):
    config = _config(
        smoke_config.retrieval, sequence_lengths=(128,), key_counts=(4,),
        query_distances=(1, 8), overwrite_counts=(1, 3),
    )
    dataset = RetrievalDataset(task=task, examples=8, config=config, seed=42, split="validation")
    for index in range(len(dataset)):
        example = dataset.example(index)
        meta = example.metadata
        tokens = example.tokens.tolist()
        assert meta["query_distance"] == meta["requested_query_distance"]
        assert meta["query_distance_definition"].startswith("requested distance anchors")
        assert len(example.answers) == (4 if task == "mqar" else 2)
        assert meta["query_count"] == len(example.answers)
        for answer in example.answers:
            query_position = answer.query_position
            value_position = answer.latest_value_position
            assert query_position is not None and value_position is not None
            query_key = tokens[query_position + 1]
            # Independent oracle over emitted tokens, one key/query at a time.
            raw_writes = [
                (position + 1, tokens[position + 1])
                for position in range(query_position)
                if tokens[position] == query_key
                and tokens[position + 1] >= config.vocabulary_size // 2
            ]
            assert query_position - value_position == answer.distance
            assert raw_writes[-1] == (value_position, answer.target)
            assert int(example.labels[answer.label_position]) == answer.target
        assert example.answers[0].distance == meta["requested_query_distance"]
        if task == "overwrite":
            latest = example.answers[0]
            assert meta["write_positions"][-1] == latest.latest_value_position
            assert meta["write_values"][-1] == latest.target
            assert len(latest.stale_values) == meta["obsolete_write_count"]
            assert set(latest.stale_values) == set(meta["write_values"][:-1])
            assert len(meta["write_positions"]) - 1 == meta["obsolete_write_count"]


def test_audited_generator_rejects_infeasible_grid_without_relabeling(smoke_config):
    config = _config(
        smoke_config.retrieval, sequence_lengths=(16,), key_counts=(4,),
        query_distances=(8,), overwrite_counts=(1,),
    )
    dataset = RetrievalDataset(task="overwrite", examples=1, config=config, seed=42, split="train")
    with pytest.raises(ValueError, match="infeasible overwrite combination"):
        dataset.example(0)


def test_audited_generator_uses_distinct_deterministic_split_domains(smoke_config):
    config = _config(
        smoke_config.retrieval, sequence_lengths=(128,), key_counts=(4,),
        query_distances=(8,), overwrite_counts=(1,),
    )
    datasets = {
        split: RetrievalDataset(task="mqar", examples=1, config=config, seed=42, split=split)
        for split in ("train", "validation", "held_out", "test")
    }
    example_tokens = {split: dataset.example(0).tokens for split, dataset in datasets.items()}
    assert all(example_tokens[split].equal(datasets[split].example(0).tokens) for split in datasets)
    assert len({tuple(tokens.tolist()) for tokens in example_tokens.values()}) == len(datasets)
