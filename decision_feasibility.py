"""Small, disposable decision fine-tuning benchmark for a Colab GPU.

This uses only the Typed Decisions training split. It is a feasibility check,
not a test-set result or the final decision-model architecture.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import shutil
import statistics
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from config import PRODUCTION_MODELS
from model import GatedDeltaNetLM


DEFAULT_CHECKPOINTS = {
    "scalar": Path("output/tinystories/30M/scalar/seed_43_ef7dde5a783e/final_checkpoint.pt"),
    "channel-wise": Path("output/tinystories/30M/channel-wise/seed_42_ced68e206c8b/final_checkpoint.pt"),
}
DRIVE_RESULTS_URL = "https://drive.google.com/file/d/1ThcNNaZssxK1lGjWw9830aU1X4G2vHCl/view"


def extract_checkpoints(archive: Path, target_dir: Path, variants: list[str]) -> dict[str, Path]:
    """Stream only the selected regular files; never extract archive paths."""
    wanted = {
        variant: tuple(DEFAULT_CHECKPOINTS[variant].parts[-5:])
        for variant in variants
    }
    found = {}
    with tarfile.open(archive, mode="r|gz") as members:
        for member in members:
            parts = PurePosixPath(member.name.replace("\\", "/")).parts
            for variant, suffix in wanted.items():
                if tuple(parts[-5:]) != suffix:
                    continue
                if variant in found:
                    raise ValueError(f"Archive contains duplicate {variant} checkpoint")
                if not member.isfile() or member.size > 1024**3:
                    raise ValueError(f"Invalid {variant} checkpoint archive member")
                source = members.extractfile(member)
                if source is None:
                    raise ValueError(f"Cannot read {variant} checkpoint archive member")
                destination = target_dir / f"{variant}_final_checkpoint.pt"
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                print(f"Extracted {variant} checkpoint ({member.size / 2**20:.1f} MiB)", flush=True)
                found[variant] = destination
    missing = set(variants) - found.keys()
    if missing:
        raise FileNotFoundError(f"Missing checkpoints in {archive}: {', '.join(sorted(missing))}")
    return found


@contextmanager
def checkpoint_paths(args, variants: list[str]):
    if args.checkpoint_archive is None and args.checkpoint_archive_url is None:
        yield {"scalar": args.scalar_checkpoint, "channel-wise": args.channel_checkpoint}
        return
    with tempfile.TemporaryDirectory(prefix="decision_feasibility_") as temporary:
        temp_dir = Path(temporary)
        archive = args.checkpoint_archive
        if args.checkpoint_archive_url is not None:
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError("Install gdown to download the checkpoint archive: pip install gdown") from exc
            archive = temp_dir / "results.tar.gz"
            print("Downloading checkpoint archive from Google Drive...", flush=True)
            downloaded = gdown.download(args.checkpoint_archive_url, str(archive), fuzzy=True, quiet=False)
            if downloaded is None or not archive.is_file():
                raise RuntimeError("Drive download failed. Check that the file is accessible to anyone with the link.")
        assert archive is not None
        if not archive.is_file():
            raise FileNotFoundError(f"Checkpoint archive not found: {archive}")
        yield extract_checkpoints(archive, temp_dir, variants)


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def choice_question(row: dict, question_id: str | None = None) -> tuple[str, dict, dict]:
    questions, gold = decoded(row["questions"]), decoded(row["gold"])
    if question_id is None:
        question_id = next(key for key, question in questions.items() if question["type"] == "choice")
    question = questions[question_id]
    if question["type"] != "choice":
        raise ValueError(f"{question_id} is not a choice question")
    return question_id, question, gold[question_id]


def prepare_examples(rows, encoder, *, question_id: str, labels: list[str], max_length: int):
    examples = []
    truncated = 0
    for row in rows:
        _, question, answer = choice_question(row, question_id)
        if sorted(question["criteria"]) != labels:
            raise ValueError("This pilot expects one fixed question and option set")
        state = decoded(row["state"])
        state_text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
        suffix = "\nQuestion: " + question["instructions"] + "\nOptions:\n"
        suffix += "\n".join(f"{label}: {question['criteria'][label]}" for label in labels)
        suffix += "\nDecision:"
        prefix_ids = encoder.encode("State: " + state_text, allowed_special="all")
        suffix_ids = encoder.encode(suffix, allowed_special="all")
        room = max_length - len(suffix_ids)
        if room < 1:
            raise ValueError("Question and options exceed max_length")
        truncated += len(prefix_ids) > room
        token_ids = (prefix_ids[:room] + suffix_ids)
        probabilities = answer["probabilities"]
        target = torch.tensor([float(probabilities[label]) for label in labels], dtype=torch.float32)
        target = target / target.sum()
        examples.append((torch.tensor(token_ids, dtype=torch.long), target))
    return examples, truncated


def make_batch(examples, indices: list[int], pad_token: int, device: torch.device):
    selected = [examples[index] for index in indices]
    lengths = torch.tensor([len(item[0]) for item in selected], device=device)
    tokens = torch.full((len(selected), int(lengths.max().item())), pad_token, dtype=torch.long)
    for index, (ids, _) in enumerate(selected):
        tokens[index, : len(ids)] = ids
    targets = torch.stack([item[1] for item in selected]).to(device)
    return tokens.to(device), lengths, targets


class DecisionPilot(nn.Module):
    def __init__(self, backbone: GatedDeltaNetLM, n_choices: int):
        super().__init__()
        # The LM output shares its weight with tok_embeddings. Remove only the
        # output module after checkpoint loading; the input embedding remains.
        del backbone.output
        self.backbone = backbone
        self.head = nn.Linear(backbone.d_model, n_choices)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor):
        hidden = self.backbone.tok_embeddings(tokens)
        for block in self.backbone.blocks:
            hidden = block(hidden)
        hidden = self.backbone.norm(hidden)
        final = hidden[torch.arange(tokens.size(0), device=tokens.device), lengths - 1]
        return self.head(final)


def load_model(path: Path, variant: str, n_choices: int) -> DecisionPilot:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}. Colab clones do not include training outputs.")
    backbone = GatedDeltaNetLM(PRODUCTION_MODELS["30M"], variant=variant, vocab_size=50_257)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(checkpoint["model_state"], strict=True)
    del checkpoint
    return DecisionPilot(backbone, n_choices)


def run_variant(args, variant, checkpoint, train_examples, val_examples, encoder, labels, device):
    torch.manual_seed(args.seed)
    model = load_model(checkpoint, variant, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(train_examples), generator=generator).tolist()
    cursor = 0
    durations = []
    losses = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train_examples), generator=generator).tolist()
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        tokens, lengths, targets = make_batch(train_examples, indices, encoder.eot_token, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(tokens, lengths)
            loss = -(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        durations.append(seconds)
        losses.append(float(loss.detach()))
        print(f"{variant} step {step}/{args.steps}: loss={losses[-1]:.4f}, seconds={seconds:.2f}", flush=True)
    model.eval()
    val_losses = []
    with torch.inference_mode():
        for index in range(0, len(val_examples), args.batch_size):
            indices = list(range(index, min(index + args.batch_size, len(val_examples))))
            tokens, lengths, targets = make_batch(val_examples, indices, encoder.eot_token, device)
            with torch.autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(tokens, lengths)
            val_losses.append(float((-(targets * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)).mean()))
    steady_steps = durations[1:] if len(durations) > 1 else durations
    result = {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "first_train_loss": losses[0],
        "last_train_loss": losses[-1],
        "mean_validation_soft_loss": statistics.mean(val_losses),
        "step_seconds": durations,
        "steady_step_median_seconds": statistics.median(steady_steps),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
    }
    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["scalar", "channel-wise", "both"], default="both")
    parser.add_argument("--scalar-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["scalar"])
    parser.add_argument("--channel-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["channel-wise"])
    archive_source = parser.add_mutually_exclusive_group()
    archive_source.add_argument("--checkpoint-archive", type=Path, help="Local results.tar.gz, including a mounted Drive path")
    archive_source.add_argument(
        "--checkpoint-archive-url", nargs="?", const=DRIVE_RESULTS_URL,
        help="Download results.tar.gz from Google Drive; omit the value to use this project's shared link",
    )
    parser.add_argument("--workflow", default="customer_service")
    parser.add_argument("--question-id", help="Choice question ID; default is first choice in the workflow")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-examples", type=int, default=48)
    parser.add_argument("--validation-examples", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("output/decision_feasibility/result.json"))
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.max_length, args.train_examples, args.validation_examples) < 1:
        parser.error("steps, batch-size, max-length, and example counts must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}", flush=True)
    rows = load_dataset("LocalLLaMA/typed-decisions", args.workflow, split="train")
    split = rows.train_test_split(test_size=0.2, seed=args.seed)
    train_rows = split["train"].select(range(min(args.train_examples, len(split["train"]))))
    val_rows = split["test"].select(range(min(args.validation_examples, len(split["test"]))))
    question_id, question, _ = choice_question(train_rows[0], args.question_id)
    labels = sorted(question["criteria"])
    encoder = tiktoken.get_encoding("gpt2")
    train_examples, train_truncated = prepare_examples(
        train_rows, encoder, question_id=question_id, labels=labels, max_length=args.max_length
    )
    val_examples, val_truncated = prepare_examples(
        val_rows, encoder, question_id=question_id, labels=labels, max_length=args.max_length
    )
    print(f"Question: {question_id}; choices: {labels}; train/val: {len(train_examples)}/{len(val_examples)}", flush=True)
    print(f"Truncated state prefixes: train={train_truncated}, validation={val_truncated}", flush=True)
    variants = ["scalar", "channel-wise"] if args.variant == "both" else [args.variant]
    with checkpoint_paths(args, variants) as paths:
        results = [run_variant(args, variant, paths[variant], train_examples, val_examples, encoder, labels, device) for variant in variants]
    report = {
        "purpose": "decision fine-tuning feasibility only; no benchmark test split used",
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "workflow": args.workflow,
        "question_id": question_id,
        "labels": labels,
        "dataset_split": "train only; 80/20 internal split",
        "checkpoint_source": (
            args.checkpoint_archive_url or str(args.checkpoint_archive)
            if args.checkpoint_archive is not None or args.checkpoint_archive_url is not None
            else "individual checkpoint paths"
        ),
        "dataset_fingerprint": rows._fingerprint,
        "available_train_cases": len(split["train"]),
        "available_validation_cases": len(split["test"]),
        "state_truncations": {"train": train_truncated, "validation": val_truncated},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
