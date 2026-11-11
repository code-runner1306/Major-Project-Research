from __future__ import annotations

import json

import pytest

from study_qualification import (
    assess_e0,
    build_e0_worker_configs,
    build_e1_worker_configs,
    build_e1_timing_worker_configs,
    build_e1_curriculum_worker_configs,
    select_e1_recipe,
    render_training_time_estimate_markdown,
    training_time_estimates,
    validate_e1_learnability,
    write_protocol_lock,
)


def _config():
    return {
        "experiment_id": "test-study",
        "conditions": [
            {"id": "scalar", "variant": "scalar", "initialization": "default"},
            {"id": "channel", "variant": "channel-wise", "initialization": "default"},
            {"id": "calibrated", "variant": "scalar", "initialization": "calibrated-geometric"},
        ],
        "datasets": ["tinystories", "mqar", "overwrite"],
        "budgets": {"10M": {"max_optimizer_steps": 10, "batch_size": 2, "grad_accum_steps": 1,
                            "context_length": 128, "validation_steps": 2, "evaluation_interval": 1,
                            "learning_rate": 0.001},
                    "30M": {"max_optimizer_steps": 20, "batch_size": 2, "grad_accum_steps": 1,
                            "context_length": 128, "validation_steps": 2, "evaluation_interval": 1,
                            "learning_rate": 0.001}},
        "qualification": {"qualification_steps": 8, "validation_interval": 2, "easy_task_length": 128,
                          "easy_task_keys": 4, "easy_task_query_distances": [8, 16],
                          "easy_overwrite_counts": [1, 2], "answer_accuracy_threshold": 0.9,
                          "consecutive_validations": 2, "learning_rate_candidates": [0.001, 0.003],
                          "extension_learning_rate": 0.01, "pilot_steps": 4},
        "retrieval": {"generator_version": "mqar-overwrite-v3", "train_examples": 100,
                      "validation_examples": 20, "held_out_examples": 20, "sequence_lengths": [128, 256],
                      "key_counts": [4, 8], "query_distances": [8, 16], "overwrite_counts": [1, 2]},
    }


def test_worker_configs_use_easy_e0_and_validation_only_e1(tmp_path):
    config = _config()
    e0 = build_e0_worker_configs(config, tmp_path)
    assert len(e0) == 4
    assert all(task["retrieval_config"]["key_counts"] == [4] for task in e0)
    assert all(task["retrieval_config"]["sequence_lengths"] == [128] for task in e0)
    assert all(task["validation_answer_accuracy"] for task in e0)
    e1 = build_e1_worker_configs(config, tmp_path)
    assert len(e1) == 12
    assert {task["learning_rate"] for task in e1} == {0.001, 0.003}
    assert all(task["retrieval_config"]["query_distances"] == [8, 16] for task in e1 if task["family"] != "tinystories")
    timing = build_e1_timing_worker_configs(config, tmp_path, 0.001)
    assert {task["scale"] for task in timing} == {"10M", "30M"}
    assert any(task["initialization"] == "calibrated-geometric" for task in timing)
    assert build_e1_curriculum_worker_configs(config, tmp_path) == []
    config["qualification"]["curriculum"] = [
        {"start_step": 1, "retrieval_config": {
            "generator_version": config["retrieval"]["generator_version"],
            "sequence_lengths": [128], "key_counts": [4], "query_distances": [8],
            "overwrite_counts": [1, 2],
        }},
        {"start_step": 3, "retrieval_config": config["retrieval"]},
    ]
    curriculum = build_e1_curriculum_worker_configs(config, tmp_path)
    assert len(curriculum) == 8
    assert all(task["curriculum"][-1]["retrieval_config"] == config["retrieval"] for task in curriculum)
    assert len({task["output_dir"] for task in curriculum}) == len(curriculum)


def test_e0_requires_consecutive_thresholds_and_e1_selects_by_validation():
    result = assess_e0([
        {"family": "mqar", "variant": "scalar", "validation_history": [
            {"answer_accuracy": 0.95}, {"answer_accuracy": 0.91}]},
        {"family": "overwrite", "variant": "channel-wise", "validation_history": [
            {"answer_accuracy": 0.89}, {"answer_accuracy": 0.95}, {"answer_accuracy": 0.92}]},
    ])
    assert result["status"] == "passed"
    failed = assess_e0([{"family": "mqar", "validation_history": [
        {"answer_accuracy": 0.95}, {"answer_accuracy": 0.89}, {"answer_accuracy": 0.95}]}])
    assert failed["status"] == "failed"

    recipe_rows = []
    for family in ("tinystories", "mqar", "overwrite"):
        for variant in ("scalar", "channel-wise"):
            for rate in (0.001, 0.003):
                value = (1.2 if rate == 0.001 else 1.3) if family == "tinystories" else 0.85
                metric = {"loss": value} if family == "tinystories" else {"answer_accuracy": value}
                recipe_rows.append({"family": family, "variant": variant, "learning_rate": rate,
                                    "metrics": {"validation_history": [metric]}})
    recipe = select_e1_recipe(recipe_rows)
    assert recipe["selected_learning_rate"] == 0.001
    assert validate_e1_learnability(recipe, 0.5)["status"] == "passed"
    recipe_rows.append({"family": "overwrite", "variant": "scalar", "learning_rate": 0.01, "status": "failed"})
    recipe = select_e1_recipe(recipe_rows)
    assert recipe["selected_learning_rate"] == 0.001
    assert len(recipe["failed_candidates"]) == 1


def test_runtime_estimate_and_protocol_lock_require_measurements(tmp_path):
    timing = training_time_estimates([
        {"family": "tinystories", "variant": "scalar", "scale": "10M", "condition_id": "C1", "metrics": {
            "training": {"fixed_optimizer_steps": 4, "wall_time_seconds": 8.0}}},
    ], {"10M": {"max_optimizer_steps": 100}})
    assert timing["budget_estimates_by_size_family_condition"]["10M"]["tinystories"]["C1"]["training_hours_per_run"] == pytest.approx(100 / 1800)
    assert timing["scenario_totals_by_size_and_seed_count"]["10M"]["5"]["combined_measured_hours"] == pytest.approx(5 * 100 / 1800)
    report = render_training_time_estimate_markdown(timing)
    assert "| Size | Seeds |" in report
    assert "training-phase wall time includes" in report.lower()
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps({"experiment_id": "test-study"}), encoding="utf-8")
    with pytest.raises(ValueError, match="E0"):
        write_protocol_lock(tmp_path, config_path, e0_result={"status": "failed"},
                            recipe={"status": "selected", "learnability_status": "passed"}, timing=timing, calibration={})
    lock = write_protocol_lock(tmp_path, config_path, e0_result={"status": "passed"},
                               recipe={"status": "selected", "learnability_status": "passed"}, timing=timing, calibration={})
    assert json.loads(lock.read_text(encoding="utf-8"))["status"] == "locked"
