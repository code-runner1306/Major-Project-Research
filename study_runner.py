"""Restartable orchestrator for the gate granularity/initialization study.

The runner owns the experiment matrix and stage ledger. Model training and
evaluation stay in isolated ``run_all.py --study-task-config`` processes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import hashlib
from typing import Any

STAGES = [f"E{i}" for i in range(7)]
FAMILIES = ("tinystories", "mqar", "overwrite")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"experiment_id", "datasets", "conditions", "sizes", "seeds", "budgets", "runtime"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Study config missing keys: {', '.join(missing)}")
    if not config["conditions"] or not config["datasets"]:
        raise ValueError("Study datasets and conditions must be non-empty")
    if len(set(config["datasets"])) != len(config["datasets"]):
        raise ValueError("Study datasets must be unique")
    if len(set(config["sizes"])) != len(config["sizes"]):
        raise ValueError("Study model sizes must be unique")
    allowed_sizes = {"10M", "30M"} if config.get("runtime", {}).get("run_mode", "production") == "production" else {"tiny"}
    if any(size not in allowed_sizes for size in config["sizes"]):
        raise ValueError(f"Invalid model size for run_mode={config.get('runtime', {}).get('run_mode', 'production')}: {config['sizes']}")
    if len({item.get("id") for item in config["conditions"]}) != len(config["conditions"]):
        raise ValueError("Condition ids must be present and unique")
    for condition in config["conditions"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(condition["id"])):
            raise ValueError(f"Unsafe condition id: {condition.get('id')!r}")
    for condition in config["conditions"]:
        if condition.get("variant") not in {"scalar", "grouped-4", "channel-wise"}:
            raise ValueError(f"Unsupported gate variant: {condition.get('variant')}")
        if condition.get("initialization") not in {"default", "calibrated-geometric"}:
            raise ValueError(f"Unsupported initialization: {condition.get('initialization')}")
    if any(f not in FAMILIES for f in config["datasets"]):
        raise ValueError("datasets must contain tinystories, mqar, and/or overwrite")
    for size in config["sizes"]:
        seeds = config["seeds"].get(size)
        if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError(f"Seed list for {size} must be non-empty and contain unique values")
        if not isinstance(config["budgets"].get(size), dict):
            raise ValueError(f"Missing training budget for configured size {size}")
    for condition in config["conditions"]:
        selected = condition.get("sizes")
        if selected and (len(set(selected)) != len(selected) or any(size not in config["sizes"] for size in selected)):
            raise ValueError(f"Condition {condition['id']} selects invalid/duplicate sizes")
    if not isinstance(config["runtime"].get("recurrence_backend"), str):
        raise ValueError("runtime.recurrence_backend must be set explicitly")
    return config


def expanded(config: dict[str, Any], root: Path, sizes: list[str] | None = None) -> list[dict[str, Any]]:
    tasks = []
    for size in sizes or config["sizes"]:
        budget = config["budgets"].get(size)
        if not isinstance(budget, dict):
            raise ValueError(f"No budget object configured for {size}")
        for family in config["datasets"]:
            for condition in config["conditions"]:
                if condition.get("sizes") and size not in condition["sizes"]:
                    continue
                for seed in config["seeds"].get(size, []):
                    cid = condition["id"]
                    task_root = root / "runs" / family / size / cid / f"seed_{seed}"
                    tasks.append({
                        "experiment_id": config["experiment_id"], "condition_id": cid,
                        "family": family, "scale": size, "variant": condition["variant"],
                        "initialization": condition["initialization"], "seed": int(seed),
                        "output_dir": str(task_root.resolve()), **budget,
                        "calibration_min_half_life": config["calibration"]["min_half_life"],
                        "calibration_max_half_life": config["calibration"]["max_half_life"],
                        "calibration_examples": config["calibration"]["examples"],
                        "retrieval_config": config.get("retrieval"),
                        "run_mode": config.get("runtime", {}).get("run_mode", "production"),
                        "recurrence_backend": config["runtime"]["recurrence_backend"],
                        "state_accumulator_dtype": config["runtime"].get("state_accumulator_dtype", "input"),
                        "recurrence_chunk_size": config["runtime"].get("recurrence_chunk_size", 64),
                    })
    return tasks


def run_logged(cmd: list[str], log_path: Path, master_log: Path, dry_run: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] START {subprocess.list2cmdline(cmd)}"
    print(line, flush=True)
    with master_log.open("a", encoding="utf-8") as master:
        master.write(line + "\n")
    if dry_run:
        return 0
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout
        for out in proc.stdout:
            print(out, end="", flush=True)
            log.write(out)
            with master_log.open("a", encoding="utf-8") as master:
                master.write(out)
        code = proc.wait()
    event = {"time": now(), "command": cmd, "return_code": code, "elapsed_seconds": time.monotonic()-start}
    with master_log.open("a", encoding="utf-8") as master:
        master.write(json.dumps(event) + "\n")
    print(f"[{now()}] {'COMPLETED' if code == 0 else 'FAILED'} rc={code} log={log_path}", flush=True)
    return code


def task_complete(task: dict[str, Any], op: str) -> bool:
    root = Path(task["output_dir"])
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        lifecycle = json.loads((root / "lifecycle.json").read_text(encoding="utf-8"))
        status = lifecycle.get("status") or lifecycle.get("state")
        identity = manifest.get("model", {}).get("study_task", manifest.get("study_task", {}))
        expected_keys = ("experiment_id", "condition_id", "family", "scale", "variant", "initialization", "seed")
        if any(identity.get(key) != task.get(key) for key in expected_keys):
            return False
        budget = identity.get("budget", identity)
        if int(budget.get("max_optimizer_steps", -1)) != int(task["max_optimizer_steps"]):
            return False
        for key in ("grad_accum_steps", "batch_size", "context_length", "validation_steps", "evaluation_interval"):
            if int(budget.get(key, -1)) != int(task[key]):
                return False
        if float(identity.get("learning_rate", -1)) != float(task["learning_rate"]):
            return False
        # Check the checkpoint payload's actual step count; a directory or lifecycle label is insufficient.
        import torch
        checkpoint = torch.load(root / "final_checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("completed_steps") != task["max_optimizer_steps"]:
            return False
        metrics_path = root / "metrics.json"
        if not metrics_path.is_file():
            return False
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training = metrics.get("training", {})
        if int(training.get("fixed_optimizer_steps", -1)) != int(task["max_optimizer_steps"]):
            return False
        if int(training.get("processed_tokens", -1)) != int(checkpoint.get("processed_tokens", -2)):
            return False
        checkpoint_metadata = metrics.get("final_checkpoint", {})
        if (not checkpoint_metadata.get("sha256")
                or int(checkpoint_metadata.get("completed_steps", -1)) != int(task["max_optimizer_steps"])
                or int(checkpoint_metadata.get("processed_tokens", -1)) != int(checkpoint.get("processed_tokens", -2))):
            return False
        digest = hashlib.sha256((root / "final_checkpoint.pt").read_bytes()).hexdigest()
        if digest != checkpoint_metadata["sha256"]:
            return False
        if op == "train":
            return status in {"trained", "complete", "evaluated"} and (root / "final_checkpoint.pt").is_file()
        return status in {"complete", "evaluated"} and (root / "metrics.json").is_file()
    except Exception:
        return False


def execute_task(task: dict[str, Any], op: str, args: argparse.Namespace, root: Path, master: Path) -> int:
    payload = {**task, "operation": op, "concurrent_training_runs": args.training_workers}
    task_file = root / "task_configs" / f"{task['family']}_{task['scale']}_{task['condition_id']}_seed_{task['seed']}_{op}.json"
    write_json(task_file, payload)
    log = root / "logs" / f"{task['family']}_{task['scale']}_{task['condition_id']}_seed_{task['seed']}_{op}.log"
    cmd = [args.python, "run_all.py", "--study-task-config", str(task_file)]
    return run_logged(cmd, log, master, args.dry_run)


def parallel_train(tasks: list[dict[str, Any]], args: argparse.Namespace, root: Path, master: Path) -> bool:
    pending: list[dict[str, Any]] = []
    missing = False
    for task in tasks:
        if args.evaluate_only:
            if not task_complete(task, "train"):
                print(f"[MISSING] training checkpoint {task['output_dir']}", flush=True)
                missing = True
                continue
        elif task_complete(task, "train"):
            print(f"[SKIPPED] validated trained checkpoint {task['output_dir']}", flush=True)
            continue
        pending.append(task)
    if missing:
        return False
    with ThreadPoolExecutor(max_workers=args.training_workers) as pool:
        active = {}
        next_task = 0
        while next_task < len(pending) and len(active) < args.training_workers:
            task = pending[next_task]
            next_task += 1
            print(f"[QUEUED] train {task['family']} {task['scale']} {task['condition_id']} seed={task['seed']}", flush=True)
            active[pool.submit(execute_task, task, "train", args, root, master)] = task
        while active:
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            failed = False
            for future in done:
                task = active.pop(future)
                if future.result() != 0:
                    print(f"[FAILED] train {task['family']} {task['scale']} {task['condition_id']} seed={task['seed']}", flush=True)
                    failed = True
            if failed:
                for future in active:
                    future.cancel()
                return False
            while next_task < len(pending) and len(active) < args.training_workers:
                task = pending[next_task]
                next_task += 1
                print(f"[QUEUED] train {task['family']} {task['scale']} {task['condition_id']} seed={task['seed']}", flush=True)
                active[pool.submit(execute_task, task, "train", args, root, master)] = task
    return True


def sequential_eval(tasks: list[dict[str, Any]], args: argparse.Namespace, root: Path, master: Path) -> bool:
    for task in tasks:
        if not task_complete(task, "train"):
            print(f"[MISSING] evaluation requires trained checkpoint {task['output_dir']}", flush=True)
            return False
        if task_complete(task, "evaluate"):
            print(f"[SKIPPED] validated evaluation {task['output_dir']}", flush=True)
            continue
        print(f"[STARTED] evaluate {task['family']} {task['scale']} {task['condition_id']} seed={task['seed']}", flush=True)
        if execute_task(task, "evaluate", args, root, master) != 0:
            return False
    return True


def run_capacity_stage(stage: str, tasks: list[dict[str, Any]], config: dict[str, Any], args: argparse.Namespace, root: Path, master: Path) -> bool:
    if stage not in {"E3", "E4"}:
        return False
    analysis_tasks = ([t for t in tasks if t["family"] in {"mqar", "overwrite"}]
                      if stage == "E3" else [t for t in tasks if t["variant"] == "channel-wise"])
    for task in analysis_tasks:
        if not task_complete(task, "evaluate"):
            print(f"[MISSING] {stage} requires completed evaluation/checkpoint: {task['output_dir']}", flush=True)
            return False
        payload = {
            "stage": stage, "experiment_id": task["experiment_id"], "condition_id": task["condition_id"],
            "family": task["family"], "scale": task["scale"], "variant": task["variant"],
            "initialization": task["initialization"], "seed": task["seed"],
            "artifact_paths": {"manifest": str(Path(task["output_dir"]) / "manifest.json"),
                               "final_checkpoint": str(Path(task["output_dir"]) / "final_checkpoint.pt"),
                               "metrics": str(Path(task["output_dir"]) / "metrics.json")},
            "retrieval_config": config["retrieval"],
            "evaluation_grid": evaluation_grid(config["retrieval"], task["family"]),
            "evaluation_examples": config["evaluation"]["examples"],
            "group_counts": config["evaluation"].get("group_counts", [8, 4, 2, 1]),
            "capacity_threshold": config["evaluation"].get("capacity_threshold", 0.9),
            "output_path": str(Path(task["output_dir"]) / f"{stage.lower()}_results_{task['scale']}.json"),
            "vocab_size": config["retrieval"].get("vocabulary_size", 8192) if task["family"] != "tinystories" else None,
            "run_mode": config.get("runtime", {}).get("run_mode", "production"),
            "metrics_path": str(Path(task["output_dir"]) / "metrics.json"),
        }
        task_file = root / "task_configs" / f"{stage}_{task['family']}_{task['scale']}_{task['condition_id']}_seed_{task['seed']}.json"
        write_json(task_file, payload)
        log = root / "logs" / f"{stage}_{task['family']}_{task['scale']}_{task['condition_id']}_seed_{task['seed']}.log"
        if run_logged([args.python, "study_analysis.py", "--task-json", str(task_file)], log, master, args.dry_run) != 0:
            return False
    return True


def evaluation_grid(retrieval: dict[str, Any], family: str) -> list[dict[str, int]]:
    grid = []
    for length in retrieval["sequence_lengths"]:
        for keys in retrieval["key_counts"]:
            for distance in retrieval["query_distances"]:
                rewrites = retrieval["overwrite_counts"] if family == "overwrite" else [1]
                for overwrite_count in rewrites:
                    if family == "mqar":
                        minimum = 5 * keys + distance - 1
                    elif keys > 1 and distance >= 3:
                        minimum = 2 * (keys + overwrite_count) + distance + 3
                    else:
                        query_tokens = 6 if keys > 1 else 3
                        minimum = 2 * (keys + overwrite_count) + distance + query_tokens - 1
                    if length >= minimum:
                        grid.append({"sequence_length": length, "key_count": keys,
                                     "query_distance": distance, "overwrite_count": overwrite_count})
    if not grid:
        raise ValueError(f"No feasible E3/E4 grid for {family}")
    return grid


def run_e6(tasks: list[dict[str, Any]], args: argparse.Namespace, root: Path, master: Path) -> bool:
    rows = []
    for task in tasks:
        directory = Path(task["output_dir"])
        rows.append({**{key: task[key] for key in ("experiment_id", "condition_id", "family", "scale", "variant", "initialization", "seed")},
                     "experiment": task["experiment_id"], "condition": task["condition_id"], "granularity": task["variant"],
                     "output_dir": task["output_dir"], "metrics_path": str(directory / "metrics.json"),
                     "manifest_path": str(directory / "manifest.json"), "checkpoint_path": str(directory / "final_checkpoint.pt"),
                     "artifact_paths": {"manifest": str(directory / "manifest.json"), "metrics": str(directory / "metrics.json"),
                                        "final_checkpoint": str(directory / "final_checkpoint.pt"), "lifecycle": str(directory / "lifecycle.json"),
                                        "e3": str(directory / f"e3_results_{task['scale']}.json"),
                                        "e4": str(directory / f"e4_results_{task['scale']}.json"),
                                        f"E3_{task['scale']}": str(directory / f"e3_results_{task['scale']}.json"),
                                        f"E4_{task['scale']}": str(directory / f"e4_results_{task['scale']}.json")}})
    payload = {"stage": "E6", "tasks": rows, "output_path": str(root / "E6_report.json")}
    task_file = root / "task_configs" / "E6_analysis.json"
    write_json(task_file, payload)
    return run_logged([args.python, "study_analysis.py", "--task-json", str(task_file)], root / "logs" / "E6.log", master, args.dry_run) == 0


def stage_record(root: Path, stage: str, status: str, detail: str = "") -> None:
    path = root / "stage_status.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        state = {"experiment_id": "", "stages": {}}
    state["stages"][stage] = {"status": status, "updated_at": now(), "detail": detail}
    write_json(path, state)


def apply_protocol_lock(tasks: list[dict[str, Any]], config: dict[str, Any], config_path: Path,
                        lock_path: Path, *, required: bool = True) -> dict[str, Any] | None:
    """Validate protocol identity and apply its selected LR/concurrency ceiling."""
    if not lock_path.is_file():
        if required:
            raise FileNotFoundError(f"Missing protocol lock {lock_path}; run E0/E1 first.")
        return None
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    smoke_scope = bool(config.get("qualification", {}).get("smoke_only"))
    expected_status, expected_scope = ("smoke_only", "smoke") if smoke_scope else ("locked", "production")
    if (lock.get("status") != expected_status or lock.get("scope") != expected_scope
            or lock.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest()):
        raise ValueError("Protocol lock status/scope/config does not match this study config.")
    rate = lock.get("recipe_selection", {}).get("selected_learning_rate")
    if not smoke_scope and rate is None:
        raise ValueError("Protocol lock does not contain a selected shared learning rate.")
    qualified_workers = int(lock.get("qualified_training_workers", 1))
    if rate is not None:
        for task in tasks:
            task["learning_rate"] = float(rate)
    curriculum = lock.get("recipe_selection", {}).get("curriculum")
    if curriculum:
        for task in tasks:
            if task["family"] not in {"mqar", "overwrite"}:
                continue
            if int(curriculum[-1]["start_step"]) > int(task["max_optimizer_steps"]):
                raise ValueError(f"Locked curriculum final stage begins after the {task['scale']} training budget")
            task["curriculum"] = curriculum
    return {**lock, "qualified_training_workers": qualified_workers}


def command_stage(stage: str, args: argparse.Namespace, root: Path, master: Path) -> bool:
    """Run existing CPU correctness path for E0; E1 records only measured pilot inputs."""
    if stage == "E0":
        out = root / "qualification" / "cpu_smoke"
        cmd = [args.python, "run_smoke.py", "--output", str(out)]
        code = run_logged(cmd, root / "logs" / "E0_cpu_smoke.log", master, args.dry_run)
        if code == 0 and not args.dry_run:
            config = load_config(Path(args.study_config).resolve())
            if config.get("qualification", {}).get("smoke_only"):
                write_json(root / "qualification" / "E0_result.json", {
                    "status": "smoke_only", "scope": "offline CPU correctness/path smoke",
                    "learnability": "not_established", "production_qualification": False,
                    "completed_at": now(),
                })
                return True
            # Main E0 additionally trains/validates both endpoints on easy tasks below.
            from study_qualification import build_e0_worker_configs, assess_e0
            primary_size = config["sizes"][0]
            results = run_qualification_workers(build_e0_worker_configs(config, root, primary_size=primary_size), args, root, master, tolerate_failures=False)
            if results is None:
                return False
            result = assess_e0(results, float(config["qualification"]["answer_accuracy_threshold"]),
                               int(config["qualification"]["consecutive_validations"]))
            write_json(root / "qualification" / "E0_result.json", result)
            return result["status"] == "passed"
        return code == 0
    if stage == "E1":
        config = load_config(Path(args.study_config).resolve())
        if config.get("qualification", {}).get("smoke_only"):
            if not args.dry_run:
                write_json(root / "recipe_selection.json", {"status": "smoke_only", "production_qualification": False})
                write_json(root / "protocol_lock.json", {"status": "smoke_only", "scope": "smoke", "experiment_id": config["experiment_id"],
                    "config_sha256": hashlib.sha256(Path(args.study_config).resolve().read_bytes()).hexdigest(), "created_at": now(),
                    "note": "This lock authorizes only the explicitly configured CPU plumbing smoke; it does not satisfy E0/E1 research qualification."})
            return True
        from study_qualification import (build_e1_worker_configs, build_e1_timing_worker_configs,
            build_e1_curriculum_worker_configs, select_e1_recipe, validate_e1_learnability, training_time_estimates, write_protocol_lock,
            render_training_time_estimate_markdown)
        # Try the planned LR range first. Failed unstable candidates are retained in logs and do not cancel peers.
        results = run_qualification_workers(build_e1_worker_configs(config, root), args, root, master, tolerate_failures=True)
        if not results:
            return False
        recipe = select_e1_recipe(results, expected_families=tuple(config["datasets"]))
        threshold = float(config["qualification"].get("minimum_final_answer_accuracy", 0.5))
        recipe["learnability"] = validate_e1_learnability(recipe, minimum_answer_accuracy=threshold)
        if recipe["learnability"].get("status") != "passed":
            extension = run_qualification_workers(build_e1_worker_configs(config, root, extension=True), args, root, master,
                                                  tolerate_failures=True)
            if extension:
                extended = select_e1_recipe(extension, expected_families=tuple(config["datasets"]))
                extended["learnability"] = validate_e1_learnability(extended, minimum_answer_accuracy=threshold)
                if extended["learnability"].get("status") == "passed":
                    recipe = extended
                else:
                    curriculum_tasks = build_e1_curriculum_worker_configs(config, root)
                    if not curriculum_tasks:
                        write_json(root / "recipe_selection.json", {"status": "failed", "reason": "Direct training and the recorded LR extension failed; no explicit curriculum was configured.",
                            "initial_candidates": recipe, "extension_candidate": extended, "minimum_final_answer_accuracy": threshold})
                        return False
                    curriculum_rows = run_qualification_workers(curriculum_tasks, args, root, master, tolerate_failures=True)
                    if not curriculum_rows:
                        return False
                    # Curriculum replaces synthetic direct-trial rows; retain TinyStories direct trials.
                    combined = [row for row in results if row.get("family") == "tinystories"] + curriculum_rows
                    curriculum_recipe = select_e1_recipe(combined, expected_families=tuple(config["datasets"]))
                    curriculum_recipe["learnability"] = validate_e1_learnability(curriculum_recipe, minimum_answer_accuracy=threshold)
                    if curriculum_recipe["learnability"].get("status") != "passed":
                        write_json(root / "recipe_selection.json", {"status": "failed", "reason": "The explicitly configured curriculum did not meet final-difficulty validation criteria.",
                            "direct_recipe": recipe, "extension_recipe": extended, "curriculum_recipe": curriculum_recipe})
                        return False
                    curriculum_recipe["curriculum"] = config["qualification"]["curriculum"]
                    recipe = curriculum_recipe
            else:
                return False
        selected_rate = float(recipe["selected_learning_rate"])
        time_workers = build_e1_timing_worker_configs(config, root, selected_rate)
        if recipe.get("curriculum"):
            for task in time_workers:
                if task["family"] in {"mqar", "overwrite"}:
                    task["curriculum"] = recipe["curriculum"]
        timing_results = run_qualification_workers(time_workers, args, root, master, tolerate_failures=False)
        if not timing_results:
            return False
        seed_counts = {size: len(config["seeds"].get(size, [])) for size in config["budgets"]}
        calibration_timings = {}
        for row in timing_results:
            try:
                manifest = json.loads((Path(row["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
                calibration_report = manifest.get("model", {}).get("calibration", {})
                elapsed = calibration_report.get("calibration_wall_time_seconds")
                if elapsed is not None:
                    calibration_timings[f"{row['scale']}/{row['family']}/{row['condition_id']}"] = float(elapsed)
            except (OSError, json.JSONDecodeError):
                continue
        timing = training_time_estimates(timing_results, config["budgets"], seed_counts=seed_counts,
                                         calibration_timings=calibration_timings)
        recipe["minimum_final_answer_accuracy"] = threshold
        recipe["final_difficulty_answer_validation_passed"] = True
        recipe["learnability_status"] = "passed"
        write_json(root / "recipe_selection.json", recipe)
        write_json(root / "training_time_estimate.json", timing)
        (root / "training_time_estimate.md").write_text(render_training_time_estimate_markdown(timing), encoding="utf-8")
        e0 = json.loads((root / "qualification" / "E0_result.json").read_text(encoding="utf-8"))
        write_protocol_lock(root, Path(args.study_config).resolve(), e0_result=e0, recipe=recipe, timing=timing,
                            calibration=config["calibration"])
        parallel_report = qualify_parallel_workers(time_workers, timing_results, args.training_workers,
                                                  config["seeds"], args, root, master)
        lock_path = root / "protocol_lock.json"
        lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        lock_payload["qualified_training_workers"] = parallel_report["qualified_training_workers"]
        lock_payload["parallel_worker_qualification"] = parallel_report
        write_json(lock_path, lock_payload)
        return True
    return False


def run_qualification_workers(worker_configs: list[dict[str, Any]], args: argparse.Namespace,
                              root: Path, master: Path, *, tolerate_failures: bool) -> list[dict[str, Any]] | None:
    """Run finite E0/E1 trials in isolated processes and return validation-only records."""
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task in worker_configs:
        task.update({
            "run_mode": "production", "recurrence_backend": load_config(Path(args.study_config).resolve())["runtime"]["recurrence_backend"],
            "calibration_min_half_life": load_config(Path(args.study_config).resolve())["calibration"]["min_half_life"],
            "calibration_max_half_life": load_config(Path(args.study_config).resolve())["calibration"]["max_half_life"],
            "concurrent_training_runs": 1,
        })
        path = root / "task_configs" / f"{task['qualification_stage']}_{task['family']}_{task['scale']}_{task['condition_id']}_{task.get('learning_rate', 'fixed')}.json"
        write_json(path, task)
        log = root / "logs" / f"{task['qualification_stage']}_{task['family']}_{task['scale']}_{task['condition_id']}_{task.get('learning_rate', 'fixed')}.log"
        code = run_logged([args.python, "run_all.py", "--study-task-config", str(path)], log, master, args.dry_run)
        if code != 0:
            failure = {**task, "status": "failed", "return_code": code, "log": str(log)}
            failures.append(failure)
            if tolerate_failures:
                completed.append(failure)
            if not tolerate_failures:
                return None
            continue
        if args.dry_run:
            continue
        run_dir = Path(task["output_dir"])
        try:
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            history_path = run_dir / "validation_history.json"
            history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.is_file() else metrics.get("validation_history", [])
            if not history:
                raise ValueError("worker completed without persisted validation history")
            completed.append({**task, "run_dir": str(run_dir), "metrics": metrics, "validation_history": history,
                              "learning_rate": task.get("learning_rate", task.get("learning_rate", 0.0006))})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failure = {**task, "status": "failed", "error": str(exc), "log": str(log)}
            failures.append(failure)
            if tolerate_failures:
                completed.append(failure)
            if not tolerate_failures:
                return None
    write_json(root / "qualification" / f"{worker_configs[0]['qualification_stage']}_failures.json", failures)
    if args.dry_run:
        return []
    return completed


def qualify_parallel_workers(tasks: list[dict[str, Any]], timing_rows: list[dict[str, Any]],
                             worker_count: int, seed_map: dict[str, list[int]], args: argparse.Namespace,
                             root: Path, master: Path) -> dict[str, Any]:
    """Measure exact concurrency for every configured size/family with isolated workers."""
    if worker_count <= 1:
        return {"qualified_training_workers": 1, "status": "sequential_default"}
    by_id = {(r["scale"], r["family"], r["condition_id"]): r for r in timing_rows}
    strata: list[dict[str, Any]] = []
    overall = True

    def run_one(task: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        safe_id = f"{task['scale']}_{task['family']}_{task['condition_id']}_seed_{task['seed']}_{task['trial_index']}"
        path = root / "task_configs" / f"parallel_{worker_count}_{safe_id}.json"
        payload = {key: value for key, value in task.items() if key != "trial_index"}
        payload["operation"] = "train"
        write_json(path, payload)
        log = root / "logs" / f"parallel_{worker_count}_{safe_id}.log"
        code = run_logged([args.python, "run_all.py", "--study-task-config", str(path)], log, master, args.dry_run)
        if code or args.dry_run:
            return code, None
        run_dir = Path(task["output_dir"])
        return 0, {**task, "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))}

    for size in seed_map:
        for family in FAMILIES:
            base = next((t for t in tasks if t["scale"] == size and t["family"] == family
                         and t["variant"] == "channel-wise" and t["initialization"] == "default"), None)
            if base is None:
                continue
            seeds = seed_map[size][:worker_count]
            if len(seeds) != worker_count:
                strata.append({"size": size, "family": family, "status": "not_qualified", "reason": "fewer distinct seeds than workers"})
                overall = False
                continue
            source = by_id.get((size, family, base["condition_id"]))
            if source is None:
                strata.append({"size": size, "family": family, "status": "not_qualified", "reason": "matching sequential timing is missing"})
                overall = False
                continue
            baseline_seconds = float(source["metrics"].get("training", {}).get("wall_time_seconds", 0.0))
            trial_tasks = []
            for index, seed in enumerate(seeds):
                trial = dict(base)
                trial.update(seed=int(seed), trial_index=index, qualification_stage=f"E1-parallel-{worker_count}",
                             concurrent_training_runs=worker_count)
                trial["output_dir"] = str((root / "qualification" / "E1" / "parallel" /
                                           f"workers_{worker_count}" / size / family / f"seed_{seed}").resolve())
                trial_tasks.append(trial)
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                outcomes = [future.result() for future in [pool.submit(run_one, task) for task in trial_tasks]]
            elapsed = time.monotonic() - started
            successes = all(code == 0 and row is not None for code, row in outcomes)
            resources = []
            tokens = 0
            per_run = {}
            for _, row in outcomes:
                if row is None:
                    continue
                metrics = row["metrics"]
                training = metrics.get("training", {})
                tokens += int(training.get("processed_tokens", 0))
                elapsed_run = float(training.get("wall_time_seconds", 0.0))
                per_run[str(row["seed"])] = {"sequential_seconds": baseline_seconds,
                                              "concurrent_seconds": elapsed_run,
                                              "slowdown_ratio": elapsed_run / baseline_seconds if baseline_seconds else None}
                capture = metrics.get("training_resources", {})
                runtime, samples = capture.get("runtime", {}), capture.get("activity_samples", [])
                gpu = [sample for sample in samples if sample.get("available") is True]
                ratios = [float(sample["memory_used_bytes"]) / float(sample["memory_total_bytes"])
                          for sample in gpu if sample.get("memory_used_bytes") is not None and sample.get("memory_total_bytes")]
                summary = capture.get("activity_summary", {})
                resources.append({"cuda_available": runtime.get("cuda_available"), "device_type": runtime.get("device_type"),
                                  "nvml_samples": len(gpu), "max_vram_used_percent": 100 * max(ratios, default=-1),
                                  "max_gpu_utilization_percent": max((float(s["gpu_utilization_percent"]) for s in gpu
                                                                        if s.get("gpu_utilization_percent") is not None), default=None),
                                  "cpu_activity_summary": {key: summary.get(key) for key in
                                      ("system_cpu_percent", "process_cpu_percent", "process_rss_bytes", "system_memory_used_bytes")}})
            telemetry_ok = len(resources) == worker_count and all(
                r["cuda_available"] and r["device_type"] == "cuda" and r["nvml_samples"] > 0
                and 0 <= r["max_vram_used_percent"] < 95 for r in resources)
            qualified = successes and telemetry_ok
            overall &= qualified
            strata.append({"size": size, "family": family, "status": "qualified" if qualified else "not_qualified",
                           "parallel_wall_seconds": elapsed, "sequential_comparison_sum_seconds": baseline_seconds * worker_count,
                           "total_processed_tokens": tokens, "measured_parallel_tokens_per_second": tokens / elapsed if elapsed else None,
                           "per_run_slowdown": per_run, "worker_telemetry": resources,
                           "reason": None if qualified else "worker failure or unavailable/unsafe NVML VRAM used/total telemetry"})
    report = {"requested_training_workers": worker_count,
              "qualified_training_workers": worker_count if overall and strata else 1,
              "status": "qualified" if overall and strata else "not_qualified", "strata": strata,
              "qualification_basis": "separate processes at each configured model size and family using channel-wise endpoint and distinct seeds",
              "cpu_utilization_interpretation": "CPU utilization is recorded but does not by itself establish CPU throttling."}
    write_json(root / "qualification" / f"parallel_workers_{worker_count}.json", report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-config", default="study_configs/granularity_init_study.json")
    p.add_argument("--output-root", default="output/granularity_init_study")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--training-workers", type=int, default=1)
    p.add_argument("--start-at", choices=STAGES, default="E0")
    p.add_argument("--stop-after", choices=STAGES)
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--size", action="append", help="limit plan to selected size; repeatable")
    args = p.parse_args()
    if args.training_workers < 1:
        p.error("--training-workers must be positive")
    if args.stop_after and STAGES.index(args.stop_after) < STAGES.index(args.start_at):
        p.error("--stop-after must be the same stage or later than --start-at")
    config_path = Path(args.study_config).resolve()
    config = load_config(config_path)
    root = Path(args.output_root).resolve()
    master = root / "logs" / "study.log"
    tasks = expanded(config, root, args.size)
    print(f"Study {config['experiment_id']}: {len(tasks)} training runs; workers={args.training_workers}; root={root}")
    for t in tasks:
        print(f"PLAN {t['family']} {t['scale']} {t['condition_id']} seed={t['seed']} steps={t['max_optimizer_steps']} output={t['output_dir']}")
    print(f"Runtime backend={config['runtime']['recurrence_backend']}; output root={root}")
    if args.dry_run:
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "expanded_plan.json", {"config": str(config_path), "tasks": tasks, "generated_at": now()})
        print("DRY RUN: expanded worker configurations, qualification configs, analysis stages, and commands (no model process is started)")
        if not config.get("qualification", {}).get("smoke_only"):
            from study_qualification import build_e0_worker_configs, build_e1_worker_configs, build_e1_timing_worker_configs
            primary_size = config["sizes"][0]
            qualification_tasks = (build_e0_worker_configs(config, root, primary_size=primary_size)
                                   + build_e1_worker_configs(config, root, primary_size=primary_size)
                                   + build_e1_timing_worker_configs(config, root, float(config["budgets"][primary_size]["learning_rate"])))
            for task in qualification_tasks:
                path = root / "task_configs" / f"{task['qualification_stage']}_{task['family']}_{task['scale']}_{task['condition_id']}_{task.get('learning_rate', 'fixed')}.json"
                print(json.dumps({"worker_config_path": str(path), "worker_config": task,
                                  "command": [args.python, "run_all.py", "--study-task-config", str(path)]}, sort_keys=True))
        for task in tasks:
            for operation in ("train", "evaluate"):
                task_file = root / "task_configs" / f"{task['family']}_{task['scale']}_{task['condition_id']}_seed_{task['seed']}_{operation}.json"
                print(json.dumps({"worker_config_path": str(task_file), "worker_config": {**task, "operation": operation,
                      "concurrent_training_runs": args.training_workers}, "command": [args.python, "run_all.py", "--study-task-config", str(task_file)]}, sort_keys=True))
        for stage in ("E3", "E4"):
            for task in tasks:
                if (stage == "E3" and task["family"] not in {"mqar", "overwrite"}) or (stage == "E4" and task["variant"] != "channel-wise"):
                    continue
                print(f"ANALYSIS PLAN {stage} {task['family']} {task['scale']} {task['condition_id']} seed={task['seed']} -> {Path(task['output_dir']) / f'{stage.lower()}_results_{task['scale']}.json'}")
        print(f"ANALYSIS PLAN E6 -> {root / 'E6_report.json'}")
        print("E0 includes the offline CPU correctness smoke; production E0 additionally measures easy-task answer-level learnability. E1 uses validation-only recipe selection and real training-time pilots.")
        return 0
    smoke_scope = bool(config.get("qualification", {}).get("smoke_only"))
    lock = None
    if args.evaluate_only or STAGES.index(args.start_at) >= STAGES.index("E2"):
        try:
            lock = apply_protocol_lock(tasks, config, config_path, root / "protocol_lock.json")
        except (OSError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.training_workers > int(lock["qualified_training_workers"]):
            print(f"Requested {args.training_workers} training workers, but E1 qualified only {lock['qualified_training_workers']}.", file=sys.stderr)
            return 2
    write_json(root / "expanded_plan.json", {"config": str(config_path), "tasks": tasks, "generated_at": now(),
               "protocol_lock": str(root / "protocol_lock.json") if (root / "protocol_lock.json").is_file() else None})
    if args.evaluate_only:
        missing = [task["output_dir"] for task in tasks if not task_complete(task, "train")]
        if missing:
            print("EVALUATE-ONLY incomplete: missing or incompatible trained checkpoints:", file=sys.stderr)
            for location in missing:
                print(f"  {location}", file=sys.stderr)
            return 2
        for family in FAMILIES:
            if not sequential_eval([t for t in tasks if t["family"] == family], args, root, master):
                return 2
        for stage, size in (("E3", "10M"), ("E4", "10M"), ("E3", "30M"), ("E4", "30M")):
            selected = [t for t in tasks if t["scale"] == size]
            if config.get("runtime", {}).get("run_mode") == "smoke" and size == "10M":
                selected = [t for t in tasks if t["scale"] == config["sizes"][0]]
            if selected and not run_capacity_stage(stage, selected, config, args, root, master):
                return 2
        return 0 if run_e6(tasks, args, root, master) else 2
    stages = STAGES[STAGES.index(args.start_at):STAGES.index(args.stop_after)+1 if args.stop_after else 7]
    for stage in stages:
        if stage in {"E0", "E1"}:
                ok = command_stage(stage, args, root, master)
                stage_record(root, stage, "complete" if ok else "incomplete", "qualification gate requires measured learnability / recipe timing")
                if not ok:
                    print(f"{stage} is incomplete; refusing to start main grid without qualification evidence.", file=sys.stderr)
                    return 2
        elif stage in {"E2", "E5"}:
                lock_path = root / "protocol_lock.json"
                if not lock_path.is_file():
                    print(f"Missing protocol lock {lock_path}; E0/E1 must pass before the core grid.", file=sys.stderr)
                    return 2
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
                expected_status = "smoke_only" if config.get("qualification", {}).get("smoke_only") else "locked"
                if lock.get("status") != expected_status or lock.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
                    print("Protocol lock is not valid for this study config; refusing core training.", file=sys.stderr)
                    return 2
                locked_rate = lock.get("recipe_selection", {}).get("selected_learning_rate")
                if expected_status == "locked" and locked_rate is None:
                    print("Protocol lock does not contain a selected shared learning rate.", file=sys.stderr)
                    return 2
                if locked_rate is not None:
                    for task in tasks:
                        task["learning_rate"] = float(locked_rate)
                size = (config["sizes"][0] if config.get("runtime", {}).get("run_mode") == "smoke" else "10M") if stage == "E2" else "30M"
                scoped = [t for t in tasks if t["scale"] == size]
                if not scoped:
                    stage_record(root, stage, "not_selected", f"No {size} conditions in selected study config")
                    print(f"[NOT SELECTED] {stage}: no {size} conditions configured.")
                    if stage == "E5":
                        continue
                    return 2
                for family in FAMILIES:
                    group = [t for t in scoped if t["family"] == family]
                    if not parallel_train(group, args, root, master):
                        stage_record(root, stage, "failed", f"training failed for {family}")
                        return 1
                    if not sequential_eval(group, args, root, master):
                        stage_record(root, stage, "failed", f"evaluation failed for {family}")
                        return 1
                stage_record(root, stage, "complete")
                if stage == "E5":
                    for post_stage in ("E3", "E4"):
                        if not run_capacity_stage(post_stage, scoped, config, args, root, master):
                            stage_record(root, f"{post_stage}-30M", "failed")
                            return 1
                        stage_record(root, f"{post_stage}-30M", "complete")
        elif stage in {"E3", "E4"}:
                base_size = config["sizes"][0] if config.get("runtime", {}).get("run_mode") == "smoke" else "10M"
                scoped = [t for t in tasks if t["scale"] == base_size]
                if not scoped:
                    stage_record(root, stage, "not_selected", "No 10M conditions configured")
                    continue
                if not run_capacity_stage(stage, scoped, config, args, root, master):
                    stage_record(root, stage, "failed")
                    return 1
                stage_record(root, stage, "complete")
        elif stage == "E6":
                if not run_e6(tasks, args, root, master):
                    stage_record(root, stage, "failed")
                    return 1
                stage_record(root, stage, "complete")
        if args.stop_after == stage:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
