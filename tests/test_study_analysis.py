from __future__ import annotations

import pytest

from study_analysis import (
    _capacity_from_grid_rows, build_e6_report, paired_interaction,
    summarize_capacity, summarize_paired_interventions,
)


def test_capacity_fixed_distance_uses_joint_cells_and_does_not_pool_marginals():
    tasks = [{
        "family": "mqar", "scale": "10M", "condition": "scalar", "seed": 42,
        "metrics": {"retrieval": {
            "key_count_by_query_distance": {
                "4": {"16": {"accuracy": 0.3}, "64": {"accuracy": 1.0}},
                "8": {"16": {"accuracy": 0.2}, "64": {"accuracy": 0.92}},
            },
            "breakdowns": {
            "key_count": {"4": {"accuracy": 0.99}, "8": {"accuracy": 0.95}},
            "query_distance": {"16": {"accuracy": 0.4}, "64": {"accuracy": 0.95}},
        }}}},
        {"family": "mqar", "scale": "10M", "condition": "channel-wise", "seed": 43,
         "metrics": {"retrieval": {"breakdowns": {
             "key_count": {"4": {"accuracy": 0.95}}, "query_distance": {"64": {"accuracy": 0.95}},
         }}}},
    ]
    result = summarize_capacity(tasks, fixed_distance=64)
    by_condition = {row["condition"]: row for row in result["summaries"]}
    scalar, channel = by_condition["scalar"], by_condition["channel-wise"]
    assert scalar["capacity_by_seed"][0]["key_count"] == 8
    assert channel["capacity_by_seed"][0]["key_count"] is None
    unavailable = next(row for row in result["seed_results"] if row["condition"] == "channel-wise")
    assert "cannot be computed" in unavailable["capacity_unavailable_reason"]


def test_e3_capacity_uses_actual_per_answer_distance_not_grid_anchor():
    result = _capacity_from_grid_rows({"grid_results": [
        {"grid": {"sequence_length": 128, "query_distance": 64, "overwrite_count": 1, "key_count": 4},
         "metrics": {"key_count_by_query_distance": {
             "4": {"12": {"correct": 9, "total": 10}, "64": {"correct": 2, "total": 10}},
         }}},
        {"grid": {"sequence_length": 128, "query_distance": 64, "overwrite_count": 1, "key_count": 8},
         "metrics": {"key_count_by_query_distance": {
             "8": {"12": {"correct": 8, "total": 10}, "64": {"correct": 1, "total": 10}},
         }}},
    ]}, threshold=.9)
    assert result["status"] == "complete"
    by_actual_distance = {row["actual_query_distance"]: row for row in result["by_fixed_difficulty"]}
    assert by_actual_distance[12]["anchor_distance"] == 64
    assert by_actual_distance[12]["largest_tested_key_count_meeting_threshold"] == 4
    assert by_actual_distance[64]["largest_tested_key_count_meeting_threshold"] is None


def test_posthoc_effects_are_paired_by_seed_against_unmodified():
    tasks = [
        {"family": "overwrite", "scale": "10M", "condition": "cw-default", "seed": 42,
         "group_count": group, "result": {"latest_value_accuracy": value}}
        for group, value in ((None, .9), (1, .7), (4, .85))
    ]
    tasks.extend([
        {"family": "overwrite", "scale": "10M", "condition": "cw-default", "seed": 43,
         "group_count": group, "result": {"latest_value_accuracy": value}}
        for group, value in ((None, .8), (1, .6), (4, .7))
    ])
    result = summarize_paired_interventions(tasks)
    one = next(item for item in result["aggregate"] if item["group_count"] == 1 and item["metric"] == "latest_value_accuracy")
    assert one["n_seeds"] == 2
    assert [row["seed"] for row in one["paired_seed_deltas"]] == [42, 43]
    assert [row["delta"] for row in one["paired_seed_deltas"]] == pytest.approx([-.2, -.2])


def test_interaction_uncertainty_uses_seed_level_difference_in_differences():
    records = []
    for seed, baseline, interaction in ((42, .5, .1), (43, .6, -.1), (44, .4, .2)):
        for init, shift in (("default", 0), ("calibrated", interaction)):
            records.extend([
                {"family": "mqar", "scale": "10M", "seed": seed, "granularity": "scalar", "initialization": init, "score": baseline},
                {"family": "mqar", "scale": "10M", "seed": seed, "granularity": "channel-wise", "initialization": init, "score": baseline + shift},
            ])
    result = paired_interaction(records, metric="score")
    assert [row["difference_in_differences"] for row in result["per_seed_effects"]] == pytest.approx([.1, -.1, .2])
    assert result["summary"][0]["n_seeds"] == 3
    assert result["unit_of_uncertainty"].startswith("paired training seed")


def test_e6_report_records_failures_and_references_artifacts_compactly():
    report = build_e6_report([
        {"family": "overwrite", "scale": "10M", "condition": "scalar", "seed": 42,
         "status": "failed", "metrics": {"retrieval": {"latest_value_accuracy": .4}},
         "metrics_path": "run/metrics.json", "checkpoint_path": "run/final_checkpoint.pt"}
    ])
    assert report["status_counts"] == {"failed": 1}
    assert report["seed_records"][0]["artifacts"] == {
        "metrics_path": "run/metrics.json", "checkpoint_path": "run/final_checkpoint.pt"}
    assert report["summary"][0]["n_seeds"] == 1
