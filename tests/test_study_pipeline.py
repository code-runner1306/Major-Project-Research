from __future__ import annotations

import json
import hashlib
import argparse
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from study_analysis import build_e6_report, run_study_task, summarize_capacity
from study_qualification import select_e1_recipe
import study_runner
from study_runner import expanded, load_config
from config import smoke_profile
from orchestration import _data_for_run
from train import SeedBundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_matrix_covers_granularity_and_initialization_cells():
    config = load_config(REPO_ROOT / "study_configs" / "granularity_init_smoke.json")

    tasks = expanded(config, Path("study-output"))

    cells = {(task["variant"], task["initialization"]) for task in tasks}
    assert cells == {
        ("scalar", "default"),
        ("grouped-4", "default"),
        ("channel-wise", "default"),
        ("scalar", "calibrated-geometric"),
        ("grouped-4", "calibrated-geometric"),
        ("channel-wise", "calibrated-geometric"),
    }
    assert len(tasks) == 12
    assert len({task["output_dir"] for task in tasks}) == len(tasks)
    assert {task["protocol_version"] for task in tasks} == {"corrected-v1"}


def test_curriculum_accepts_step_one_and_increasing_stages(tmp_path: Path):
    profile = smoke_profile(tmp_path / "curriculum")
    profile = replace(
        profile,
        budget=replace(profile.budget, max_optimizer_steps=3),
        curriculum=(
            {"start_step": 1, "retrieval_config": {"key_counts": [2]}},
            {"start_step": 2, "retrieval_config": {"key_counts": [3]}},
        ),
    )

    *_, curriculum_loaders = _data_for_run(
        profile, "mqar", SeedBundle.from_research_seed(7), tmp_path / "curriculum-run"
    )

    assert [step for step, _loader in curriculum_loaders] == [1, 2]


def test_dry_run_expands_worker_commands_without_running_them(tmp_path: Path):
    output_root = tmp_path / "dry-run"
    completed = subprocess.run(
        [
            sys.executable,
            "study_runner.py",
            "--study-config",
            "study_configs/granularity_init_smoke.json",
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DRY RUN: expanded worker configurations, qualification configs, analysis stages, and commands" in completed.stdout
    plan = json.loads((output_root / "expanded_plan.json").read_text(encoding="utf-8"))
    assert len(plan["tasks"]) == 12
    assert not (output_root / "runs").exists()
    assert not (output_root / "logs").exists()


def test_capacity_at_fixed_distance_uses_joint_breakdown_not_marginals():
    metrics = {
        "retrieval": {
            "breakdowns": {
                # Marginal key-count results would report capacity 8, but at
                # distance 64 only key count 2 reaches the threshold.
                "key_count": {
                    "2": {"accuracy": 1.0, "correct": 10, "total": 10},
                    "4": {"accuracy": 0.95, "correct": 19, "total": 20},
                    "8": {"accuracy": 0.92, "correct": 23, "total": 25},
                },
                "query_distance": {"16": {"accuracy": 1.0}, "64": {"accuracy": 0.8}},
                "key_count_by_query_distance": {
                    "2": {"16": {"accuracy": 1.0, "correct": 10, "total": 10}, "64": {"accuracy": 0.95, "correct": 19, "total": 20}},
                    "4": {"16": {"accuracy": 0.98, "correct": 49, "total": 50}, "64": {"accuracy": 0.8, "correct": 8, "total": 10}},
                    "8": {"16": {"accuracy": 0.92, "correct": 23, "total": 25}, "64": {"accuracy": 0.4, "correct": 2, "total": 5}},
                },
            }
        }
    }

    result = summarize_capacity(
        [{"family": "mqar", "scale": "tiny", "condition": "scalar", "seed": 7, "metrics": metrics}],
        accuracy_threshold=0.9,
        fixed_distance=64,
    )

    seed = result["seed_results"][0]
    assert seed["capacity_key_count"] == 2
    assert seed["observed"] == [
        {"key_count": 2, "accuracy": 0.95, "correct": 19, "total": 20},
        {"key_count": 4, "accuracy": 0.8, "correct": 8, "total": 10},
        {"key_count": 8, "accuracy": 0.4, "correct": 2, "total": 5},
    ]


def test_capacity_at_fixed_distance_reports_unavailable_without_joint_data():
    metrics = {"retrieval": {"breakdowns": {"key_count": {"2": {"accuracy": 1.0}}}}}

    result = summarize_capacity(
        [{"family": "mqar", "scale": "tiny", "condition": "scalar", "seed": 7, "metrics": metrics}],
        fixed_distance=64,
    )

    seed = result["seed_results"][0]
    assert seed["capacity_key_count"] is None
    assert "distance-specific" in seed["capacity_unavailable_reason"]


def test_e6_marks_empty_or_missing_artifacts_incomplete(tmp_path: Path):
    empty = build_e6_report([])
    assert empty["status"] == "incomplete"

    output = tmp_path / "E6_report.json"
    task = {
        "stage": "E6",
        "output_path": str(output),
        "tasks": [{
            "family": "mqar",
            "scale": "tiny",
            "condition": "scalar",
            "variant": "scalar",
            "initialization": "default",
            "seed": 7,
            "artifact_paths": {
                "manifest": str(tmp_path / "missing_manifest.json"),
                "metrics": str(tmp_path / "missing_metrics.json"),
                "final_checkpoint": str(tmp_path / "missing_checkpoint.pt"),
                "lifecycle": str(tmp_path / "missing_lifecycle.json"),
                "E3_tiny": str(tmp_path / "missing_E3.json"),
            },
        }],
    }

    with pytest.raises(RuntimeError, match="E6 analysis incomplete"):
        run_study_task(task)

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["status"] == "incomplete"
    assert persisted["summary"] == []
    assert persisted["exclusions"]


def test_e1_selects_lower_lm_validation_loss_within_family():
    records = [
        {"family": "tinystories", "variant": variant, "learning_rate": rate,
         "metrics": {"validation_history": [{"step": 2, "loss": loss}]}}
        for variant in ("scalar", "channel-wise")
        for rate, loss in ((0.001, 2.0), (0.003, 1.0))
    ]

    recipe = select_e1_recipe(records, expected_families=("tinystories",))

    assert recipe["selected_learning_rate"] == 0.003
    assert recipe["family_validation_scores_by_learning_rate"]["tinystories"]["0.003"] > recipe[
        "family_validation_scores_by_learning_rate"]["tinystories"]["0.001"
    ]


def test_protocol_locked_e1_rate_reaches_core_train_and_eval_tasks(tmp_path: Path, monkeypatch):
    config_path = REPO_ROOT / "study_configs" / "granularity_init_smoke.json"
    output_root = tmp_path / "locked-rate"
    output_root.mkdir()
    lock = {
        "status": "smoke_only",
        "scope": "smoke",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "recipe_selection": {"selected_learning_rate": 0.003},
    }
    (output_root / "protocol_lock.json").write_text(json.dumps(lock), encoding="utf-8")
    seen_train_rates = []
    seen_eval_rates = []

    def record_train(tasks, *_args):
        seen_train_rates.extend(task["learning_rate"] for task in tasks)
        return True

    def record_eval(tasks, *_args):
        seen_eval_rates.extend(task["learning_rate"] for task in tasks)
        return True

    monkeypatch.setattr(study_runner, "parallel_train", record_train)
    monkeypatch.setattr(study_runner, "sequential_eval", record_eval)
    monkeypatch.setattr(sys, "argv", [
        "study_runner.py", "--study-config", str(config_path), "--output-root", str(output_root),
        "--start-at", "E2", "--stop-after", "E2",
    ])

    assert study_runner.main() == 0
    assert seen_train_rates and set(seen_train_rates) == {0.003}
    assert seen_eval_rates and set(seen_eval_rates) == {0.003}


def test_parallel_qualification_strips_internal_trial_fields_from_worker_json(tmp_path: Path, monkeypatch):
    captured = []

    def fake_run_logged(command, _log_path, _master_log, _dry_run):
        task = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        captured.append(task)
        run_dir = Path(task["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps({"training": {"wall_time_seconds": 1.0, "processed_tokens": 10},
                        "training_resources": {"runtime": {"cuda_available": False, "device_type": "cpu"},
                                                "activity_samples": [], "activity_summary": {}}}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(study_runner, "run_logged", fake_run_logged)
    args = argparse.Namespace(python=sys.executable, dry_run=False)
    base = {
        "experiment_id": "qualification-test", "condition_id": "channel-default",
        "family": "mqar", "scale": "tiny", "variant": "channel-wise",
        "initialization": "default", "seed": 42, "output_dir": str(tmp_path / "base"),
        "max_optimizer_steps": 2, "batch_size": 2, "grad_accum_steps": 1,
        "context_length": 16, "validation_steps": 1, "evaluation_interval": 1,
        "learning_rate": 0.001, "run_mode": "smoke",
    }
    report = study_runner.qualify_parallel_workers(
        [base],
        [{"scale": "tiny", "family": "mqar", "condition_id": "channel-default",
          "metrics": {"training": {"wall_time_seconds": 1.0}}}],
        worker_count=2,
        seed_map={"tiny": [42, 43]},
        args=args,
        root=tmp_path,
        master=tmp_path / "study.log",
    )

    assert len(captured) == 2
    assert all("trial_index" not in task for task in captured)
    assert all(task["operation"] == "train" for task in captured)
    assert report["qualified_training_workers"] == 1  # CPU has no NVML VRAM qualification.
