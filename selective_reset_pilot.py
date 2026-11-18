"""Four-run selective-reset pilot. Run from the repository root with Python."""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "third_party" / "dualgoose"))
from dualgoose.model import build_tiny_denoiser

# Input alphabet: filler 0, NOOP 1, reset red/blue 2..3, workspace 4..5,
# local keys 6..37, values 38..69. EMPTY is a reserved answer.
EMPTY, VOCAB, LENGTH = 70, 71, 64
LOADS = (1, 2, 4, 8)
QUERIES = ("target", "survivor", "rewrite", "absent")


def episodes(rng, batch, load):
    """Produce independent workspaces; key names can overlap between workspaces."""
    row = np.arange(batch)
    target = rng.integers(2, size=batch)
    names = rng.random((batch, 2, 32)).argsort(axis=2) + 6
    slot = np.arange(16)
    groups = np.where(slot < load, target[:, None], 1 - target[:, None])
    local = np.where(slot < load, slot, slot - load)
    keys = names[row[:, None], groups, local]
    values = rng.integers(38, 70, (batch, 16))
    triples = np.stack((4 + groups, keys, values), axis=2)
    order = rng.random((batch, 16)).argsort(axis=1)
    prefix = np.zeros((batch, LENGTH), dtype=np.int64)
    start = rng.integers(4, size=batch)
    prefix[row[:, None], start[:, None] + np.arange(48)] = triples[row[:, None], order].reshape(batch, 48)
    event = start + 48 + rng.integers(4, size=batch)
    reset_key = rng.integers(load, size=batch)
    survivor = rng.integers(load, 16, size=batch)
    stale = values[row, reset_key]
    fresh = 38 + (stale - 38 + rng.integers(1, 32, size=batch)) % 32
    missing_group = rng.integers(2, size=batch)
    missing_key = names[row, missing_group, 31]
    queries = np.stack((keys[row, reset_key], keys[row, survivor], keys[row, reset_key], missing_key))
    spaces = np.stack((target, 1 - target, target, missing_group))
    answers = np.stack((stale, values[row, survivor], fresh, np.full(batch, EMPTY)))
    # Include a post-event survivor update in every branch. Randomize its order
    # relative to target rewrites so the newest value is not always the answer.
    decoy = load + (survivor - load + 1) % (16 - load)
    offset = rng.choice([1, 4], size=batch)
    prefix[row, event + offset] = 5 - target
    prefix[row, event + offset + 1] = keys[row, decoy]
    prefix[row, event + offset + 2] = rng.integers(38, 70, size=batch)
    return prefix, event, target, queries, spaces, answers


def branch(ep, reset, kind):
    prefix, event, target, keys, spaces, answers = ep
    x = prefix.copy()
    row = np.arange(len(x))
    x[row, event] = np.where(reset, 2 + target, 1)
    rewrite = np.broadcast_to(kind == 2, row.shape)
    r = row[rewrite]
    position = event[r] + np.where(prefix[r, event[r] + 1] == 0, 1, 4)
    x[r, position] = 4 + target[r]
    x[r, position + 1] = keys[2, r]
    x[r, position + 2] = answers[2, r]
    x[:, -2] = 4 + spaces[kind, row]
    x[:, -1] = keys[kind, row]
    y = np.where(reset & (kind == 0), EMPTY, answers[kind, row])
    return x, y


def audit():
    rng = np.random.default_rng(3317)
    for load in LOADS:
        ep = episodes(rng, 128, load)
        for kind in range(4):
            normal, _ = branch(ep, False, kind)
            for reset in (True, False):
                x, labels = branch(ep, reset, kind)
                assert np.all((x != normal).sum(axis=1) == int(reset))
                for tokens, label in zip(x, labels):
                    memory = {}
                    pos = 0
                    while pos < LENGTH - 2:
                        token = tokens[pos]
                        if token in (4, 5):
                            memory[token, tokens[pos + 1]] = tokens[pos + 2]
                            pos += 3
                        else:
                            if token in (2, 3):
                                memory = {k: v for k, v in memory.items() if k[0] != token + 2}
                            pos += 1
                    assert memory.get(tuple(tokens[-2:]), EMPTY) == label


def logits(model, x, device):
    tokens = torch.from_numpy(x).to(device)
    return model(tokens, torch.ones(len(x), device=device))[:, -1]


@torch.no_grad()
def evaluate(model, args):
    rng = np.random.default_rng(908731)
    result = {}
    model.eval()
    for load in LOADS:
        counts = {}
        for _ in range(args.eval_batches):
            ep = episodes(rng, args.batch, load)
            for reset in (False, True):
                tag = "reset" if reset else "noop"
                correct = []
                for kind, name in enumerate(QUERIES):
                    x, y = branch(ep, reset, kind)
                    pred = logits(model, x, args.device).argmax(1).cpu().numpy()
                    correct.append(pred == y)
                    key = f"{tag}_{name}"
                    counts[key] = counts.get(key, 0) + int(correct[-1].sum())
                    if reset and kind == 0:
                        key = "stale_value_returned"
                        counts[key] = counts.get(key, 0) + int((pred == ep[-1][0]).sum())
                key = f"{tag}_joint"
                counts[key] = counts.get(key, 0) + int((correct[0] & correct[1]).sum())
        result[load] = {key: n / (args.batch * args.eval_batches) for key, n in counts.items()}
    return result


def train(args, seed, family, folder):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build_tiny_denoiser(VOCAB, LENGTH, model_type=family, d_model=32,
                                n_layers=2, n_heads=1, short_conv=4, direction="forward",
                                merge="static").to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=0.0001)
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        ep = episodes(rng, args.batch, rng.choice(LOADS))
        x, y = branch(ep, rng.integers(2, size=args.batch).astype(bool),
                      rng.integers(4, size=args.batch))
        loss = F.cross_entropy(logits(model, x, args.device), torch.from_numpy(y).long().to(args.device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step == 1 or step % 200 == 0 or step == args.steps:
            print(f"{family} seed={seed} step={step} loss={loss.item():.4f}", flush=True)
    elapsed = time.perf_counter() - start
    stem = folder / f"{family}_{seed}"
    torch.save(model.state_dict(), stem.with_suffix(".pt"))
    report = dict(seed=seed, family=family, settings=vars(args), training_seconds=elapsed,
                  parameters=sum(p.numel() for p in model.parameters()), accuracy=evaluate(model, args))
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2))


def train_seed(args, seed, folder, device):
    args.device = device
    print(f"Worker seed={seed} device={device}", flush=True)
    for family in ("gated_deltanet_scalar", "gated_deltanet"):
        train(args, seed, family, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--out", default="results/selective_reset_pilot")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_batches) < 1:
        parser.error("steps, batch and eval-batches must be positive")
    audit()
    devices = [args.device, args.device]
    if torch.device(args.device).type == "cuda":
        available = torch.cuda.device_count()
        if not available:
            parser.error("CUDA unavailable. Enable a Kaggle GPU or pass --device cpu.")
        if args.device == "cuda":
            devices = ["cuda:0", f"cuda:{1 if available > 1 else 0}"]
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        jobs = [pool.submit(train_seed, args, seed, folder, device)
                for seed, device in zip((42, 43), devices)]
        for job in jobs:
            job.result()
