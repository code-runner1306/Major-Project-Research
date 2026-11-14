"""Run exactly two selected study models on two GPUs and export their evidence."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parent
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".pkl", ".pickle"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--condition", choices=["C1", "C3", "C4", "C6"],
                           help="Run MQAR and overwrite for this condition")
    selection.add_argument("--models", nargs=2, metavar="FAMILY:CONDITION",
                           help="Exactly two selections, e.g. mqar:C1 overwrite:C1")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--study-config", type=Path, default=ROOT / "study_configs/granularity_corrected_pilot.json")
    p.add_argument("--output-root", type=Path)
    p.add_argument("--hours", type=float, default=12)
    p.add_argument("--reserve-minutes", type=int, default=20)
    p.add_argument("--approved-by", help="Record the person authorizing these two selected runs")
    p.add_argument("--push-results", action="store_true", help="Push evidence to a unique results branch")
    p.add_argument("--results-remote", default="origin", help="Configured Git remote name")
    p.add_argument("--dry-run", action="store_true")
    return p


def selections(args, study):
    conditions = {c["id"].split("-")[0]: c for c in study["conditions"]}
    names = args.models or [f"mqar:{args.condition}", f"overwrite:{args.condition}"]
    result = []
    for name in names:
        family, sep, condition = name.partition(":")
        if not sep or family not in ("mqar", "overwrite") or condition not in conditions:
            raise ValueError(f"Invalid model {name!r}; use mqar|overwrite:C1|C3|C4|C6")
        result.append((family, conditions[condition]))
    if names[0] == names[1]:
        raise ValueError("Select two different models")
    return result


def task_config(study, family, condition, seed, output):
    budget = study["budgets"]["10M"]
    retrieval = dict(study["retrieval"], validation_examples=512)
    return {
        "operation": "train", "experiment_id": study["experiment_id"],
        "condition_id": condition["id"], "family": family, "scale": "10M",
        "variant": condition["variant"], "initialization": condition["initialization"],
        "seed": seed, "output_dir": str(output),
        **{key: budget[key] for key in ("max_optimizer_steps", "batch_size", "grad_accum_steps",
                                       "context_length", "validation_steps", "evaluation_interval", "learning_rate")},
        "calibration_min_half_life": study["calibration"]["min_half_life"],
        "calibration_max_half_life": study["calibration"]["max_half_life"],
        "calibration_examples": study["calibration"]["examples"],
        "retrieval_config": retrieval, "run_mode": "production",
        "recurrence_backend": study["runtime"].get("recurrence_backend", "reference"),
        "state_accumulator_dtype": study["runtime"]["state_accumulator_dtype"],
        "recurrence_chunk_size": 64, "validation_answer_accuracy": True,
        "concurrent_training_runs": 2, "protocol_version": study["runtime"]["protocol_version"],
    }


def run_phase(tasks, phase, output, deadline):
    processes, logs, starts, results = {}, {}, {}, {}
    try:
        for gpu, task in enumerate(tasks):
            label = f"{task['family']}_{task['condition_id']}"
            config = dict(task, operation=phase)
            path = output / "tasks" / f"{label}_{phase}.json"
            write_json(path, config)
            logs[label] = (output / "logs" / f"{label}_{phase}.log").open("w", encoding="utf-8")
            starts[label] = time.monotonic()
            processes[label] = subprocess.Popen(
                [sys.executable, str(ROOT / "run_all.py"), "--study-task-config", str(path)],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                stdout=logs[label], stderr=subprocess.STDOUT, start_new_session=True,
            )
            print(f"{phase}: {label} on GPU {gpu}; log={logs[label].name}", flush=True)
        while len(results) < len(processes):
            if time.monotonic() >= deadline:
                raise TimeoutError("Session time reserve reached")
            for label, process in processes.items():
                if label not in results and process.poll() is not None:
                    results[label] = {"exit_code": process.returncode,
                                      "elapsed_seconds": time.monotonic() - starts[label]}
                    write_json(output / f"{phase}_status.json", results)
                    print(f"{phase}: {label}: {results[label]}", flush=True)
            time.sleep(1)
        return results
    finally:
        # Stop only this launcher's subprocess groups when the deadline expires.
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for log in logs.values():
            log.close()


def checkpoint_path(path):
    return (path.suffix.lower() in CHECKPOINT_SUFFIXES
            or any("checkpoint" in part.lower() for part in path.parts[:-1])
            or ("checkpoint" in path.name.lower() and not path.suffix))


def export_evidence(output):
    """Copy evidence into a sibling directory; preserve weights only in raw output."""
    bundle = output.with_name(output.name + "_results")
    bundle.mkdir(exist_ok=False)
    excluded, included = [], []
    for source in sorted(output.rglob("*")):
        relative = source.relative_to(output)
        if source.is_symlink():
            excluded.append(str(relative))
        elif source.is_file():
            if checkpoint_path(relative):
                excluded.append(str(relative))
                continue
            destination = bundle / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            included.append({"path": str(relative), "bytes": source.stat().st_size,
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    write_json(bundle / "export_manifest.json", {"included": included, "excluded": excluded})
    return bundle


def remote_url(name):
    if name.startswith("-"):
        raise ValueError("Invalid remote name")
    url = subprocess.check_output(["git", "remote", "get-url", name], cwd=ROOT, text=True).strip()
    # Never write a credential-bearing remote URL into reports or Git config.
    if url.startswith("https://") and "@" in url.split("/", 3)[2]:
        raise ValueError("Use a credential-free remote URL with Git credentials or GH_TOKEN")
    return url


@contextmanager
def git_environment():
    with tempfile.TemporaryDirectory(prefix="kaggle-git-auth-") as directory:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_SSH_COMMAND="ssh -o BatchMode=yes")
        if os.environ.get("GH_TOKEN"):
            helper = Path(directory) / "askpass.sh"
            helper.write_text('#!/bin/sh\ncase "$1" in *Username*) printf "%s" "x-access-token" ;; *) printf "%s" "$GH_TOKEN" ;; esac\n')
            helper.chmod(0o700)
            env["GIT_ASKPASS"] = str(helper)
        yield env


def publish(bundle, url, branch):
    """Use a separate, orphan result branch so notebook pushes cannot conflict."""
    with tempfile.TemporaryDirectory(prefix="kaggle-publish-") as directory, git_environment() as env:
        staging = Path(directory)
        shutil.copytree(bundle, staging / "results")
        def git(*arguments):
            subprocess.run(["git", *arguments], cwd=staging, env=env, check=True, timeout=300)
        git("init", "-b", branch)
        git("config", "user.name", "Kaggle research results")
        git("config", "user.email", "kaggle-results@users.noreply.github.com")
        git("add", "--", "results")
        git("commit", "-m", f"Record {bundle.name} (checkpoints excluded)")
        git("push", url, f"HEAD:refs/heads/{branch}")


def main(argv=None):
    args = parser().parse_args(argv)
    study = json.loads(args.study_config.read_text(encoding="utf-8-sig"))
    if study["runtime"].get("protocol_version") != "corrected-v1" or study["retrieval"].get("generator_version") != "mqar-overwrite-v4":
        raise ValueError("This launcher requires corrected-v1 and mqar-overwrite-v4")
    if args.hours <= 0 or not 0 < args.reserve_minutes < args.hours * 60:
        raise ValueError("Reserve must be positive and smaller than the session duration")
    if not 0 <= args.seed < 2**32:
        raise ValueError("Seed must be between 0 and 2**32-1")
    selected = selections(args, study)
    session = f"seed{args.seed}_" + "_".join(f"{f}-{c['id'].split('-')[0]}" for f, c in selected)
    session += "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = (args.output_root or Path("/kaggle/working") / session).resolve()
    tasks = [task_config(study, f, c, args.seed, output / "runs" / f / c["id"] / f"seed_{args.seed}") for f, c in selected]
    plan = {"workers": 2, "gpu_assignment": [0, 1], "tasks": tasks,
            "hours": args.hours, "reserve_minutes": args.reserve_minutes,
            "results_branch": "kaggle-results/" + session if args.push_results else None}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return 0
    if os.name != "posix":
        raise RuntimeError("Production execution requires Linux (Kaggle); --dry-run is portable")
    if not args.approved_by or not args.approved_by.strip():
        raise ValueError("Supply --approved-by to record authorization for the selected two-model session")
    authorized = dict(study, compute_authorization={"status": "approved", "approved_by": args.approved_by,
        "approved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": {"seed": args.seed, "models": [f"{f}:{c['id']}" for f, c in selected], "steps": study["budgets"]["10M"]["max_optimizer_steps"]}})
    from study_runner import require_compute_authorization
    require_compute_authorization(authorized, stage="selected-kaggle-session")
    url = remote_url(args.results_remote) if args.push_results else None
    if args.push_results:
        # Check authentication and receive access before spending training time.
        # --dry-run does not create a branch or publish source files.
        with git_environment() as env:
            subprocess.run(["git", "push", "--dry-run", url,
                            f"HEAD:refs/heads/{plan['results_branch']}"],
                           cwd=ROOT, env=env, check=True, timeout=60)
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError("Two visible CUDA GPUs are required")
    hardware = []
    for gpu in (0, 1):
        with torch.cuda.device(gpu):
            free, total = torch.cuda.mem_get_info()
        if free < 6 * 1024**3:
            raise RuntimeError(f"GPU {gpu} has less than 6 GiB free")
        hardware.append({"gpu": gpu, "name": torch.cuda.get_device_name(gpu), "free_bytes": free, "total_bytes": total})
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    write_json(output / "plan.json", plan)
    write_json(output / "authorized_study_config.json", authorized)
    write_json(output / "hardware.json", hardware)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(output / "source.json", {"revision": revision, "config_sha256": hashlib.sha256(args.study_config.read_bytes()).hexdigest()})
    start = time.monotonic()
    deadline = start + args.hours * 3600 - args.reserve_minutes * 60
    summary = {"status": "running", "source_revision": revision, "models": 2, "seed": args.seed}
    code = 0
    try:
        summary["train"] = run_phase(tasks, "train", output, deadline)
        if any(row["exit_code"] for row in summary["train"].values()):
            raise RuntimeError("Training failed; see per-model logs")
        summary["evaluate"] = run_phase(tasks, "evaluate", output, deadline)
        if any(row["exit_code"] for row in summary["evaluate"].values()):
            raise RuntimeError("Evaluation failed; see per-model logs")
        summary["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        summary.update(status="incomplete", error=str(error))
        code = 1
    finally:
        summary["elapsed_seconds"] = time.monotonic() - start
        write_json(output / "session_summary.json", summary)
        bundle = export_evidence(output)
        print(f"Raw artifacts (including checkpoints): {output}\nEvidence bundle: {bundle}", flush=True)
        if args.push_results:
            try:
                publish(bundle, url, plan["results_branch"])
                print(f"Published results branch: {plan['results_branch']}", flush=True)
            except Exception as error:
                print(f"RESULT PUSH FAILED: {error}. Evidence remains at {bundle}", file=sys.stderr)
                code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
