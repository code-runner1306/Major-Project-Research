"""[Major-Project-Research addition] E6: decay interventions on trained channel-wise checkpoints.

For each saved checkpoint of a channel-wise gated arm (armed_gdn, gated_deltanet, armed_gdn_fwd), re-evaluate with the
decay of every gated recurrence replaced at evaluation time:
  original               no change (must reproduce the run's recorded eval accuracy: a built-in check)
  fixed_channel_mean     each channel's decay is its mean over calibration batches (keeps channel diversity,
                         removes input dependence)
  token_channel_average  each token's channel decays are replaced by their average (keeps input dependence,
                         removes channel diversity)
  fixed_scalar_mean      one constant for all channels and tokens (removes both)
Calibration batches come from the training generator with a separate seed; evaluation uses the exact generator and
seed of the original run (seed + 10000), so "original" reproduces the recorded accuracy.

  python scripts/decay_interventions.py --kind mqar --checkpoints checkpoints/e1/*.pt
  python scripts/decay_interventions.py --kind long --checkpoints checkpoints/e5/*.pt
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import long_retrieval_probe as lrp  # noqa: E402
import mqar_probe as mqp  # noqa: E402
from dualgoose.diffusion import apply_subs_sample  # noqa: E402
from dualgoose.mixers import DeltaNetRefLayer  # noqa: E402
from dualgoose.toy_data import make_long_retrieval_batch, make_mqar_batch  # noqa: E402

OUTDIR = Path("results/decay_interventions")
CHANNEL_ARMS = {"armed_gdn", "gated_deltanet", "armed_gdn_fwd"}


def batches(kind, meta, n, batch, generator, device, training_like):
    vocab, mask_id, seq_len = meta["vocab"], meta["vocab"] - 1, meta["seq_len"]
    for _ in range(n):
        if kind == "mqar":
            yield make_mqar_batch(batch, seq_len, vocab, mask_id, pairs=meta["K"], query_layout="after_table",
                                  device=device, generator=generator)
        else:
            gap = None
            if training_like and meta["curriculum"]:  # uniform curriculum, as in training
                max_gap = seq_len - 2 * meta["pairs"] - 2 * meta["n_queries"]
                gap = int(torch.randint(0, max_gap + 1, (1,), device=device, generator=generator))
            yield make_long_retrieval_batch(batch, seq_len, vocab, mask_id, pairs=meta["pairs"],
                                            n_queries=meta["n_queries"], gap=gap, collision=meta["collision"],
                                            device=device, generator=generator)


def accuracy(model, kind, meta, args, device):
    egen = torch.Generator(device=device).manual_seed(meta["seed"] + 10_000)
    correct = total = 0
    with torch.no_grad():
        for b in batches(kind, meta, args.eval_iters, args.eval_batch, egen, device, training_like=False):
            pred = apply_subs_sample(model(b.corrupted, b.t), b.corrupted, meta["vocab"] - 1)
            m = b.target_mask
            correct += int(pred[m].eq(b.clean[m]).sum().cpu())
            total += int(m.sum().cpu())
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["mqar", "long"], required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--eval-batch", type=int, default=None, help="default: the probe's own eval batch")
    ap.add_argument("--eval-iters", type=int, default=None, help="default: the probe's own eval iterations")
    ap.add_argument("--calib-iters", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.eval_batch = args.eval_batch or (512 if args.kind == "mqar" else 256)
    args.eval_iters = args.eval_iters or (4 if args.kind == "mqar" else 8)
    device = torch.device(args.device)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    paths = sorted({p for pattern in args.checkpoints for p in glob.glob(pattern)})
    for path in paths:
        meta = torch.load(path, map_location="cpu", weights_only=False)
        if meta["label"] not in CHANNEL_ARMS:
            print(f"SKIP {path}: not a channel-wise arm", flush=True)
            continue
        probe = mqp if args.kind == "mqar" else lrp
        model = probe.build(meta["label"], meta["vocab"], meta["seq_len"], meta["d_model"], meta["layers"],
                            meta["heads"], meta.get("mamba2_d_state")).to(device)
        model.load_state_dict(meta["model"], strict=True)
        model.eval()
        cells = [m for m in model.modules() if isinstance(m, DeltaNetRefLayer) and m.gated]
        sums = {id(c): torch.zeros(c.decay_width, device=device) for c in cells}
        counts = {id(c): 0 for c in cells}

        def recorder(cell):
            def hook(w):
                sums[id(cell)] += w.float().sum(dim=(0, 1))
                counts[id(cell)] += w.shape[0] * w.shape[1]
                return w
            return hook
        for c in cells:
            c.decay_override = recorder(c)
        cgen = torch.Generator(device=device).manual_seed(meta["seed"] + 20_000)
        with torch.no_grad():
            for b in batches(args.kind, meta, args.calib_iters, args.eval_batch, cgen, device, training_like=True):
                model(b.corrupted, b.t)
        means = {i: sums[i] / max(1, counts[i]) for i in sums}
        interventions = {
            "original": lambda c: None,
            "fixed_channel_mean": lambda c: (lambda w, m=means[id(c)]: m.to(w.dtype).expand_as(w)),
            "token_channel_average": lambda c: (lambda w: w.mean(dim=-1, keepdim=True).expand_as(w)),
            "fixed_scalar_mean": lambda c: (lambda w, m=means[id(c)].mean(): m.to(w.dtype).expand_as(w)),
        }
        result = dict(checkpoint=str(path), label=meta["label"], seed=meta["seed"], kind=args.kind,
                      calibration_batches=args.calib_iters, eval_batch=args.eval_batch, eval_iters=args.eval_iters,
                      decay_channel_means={f"cell{j}": [round(v, 5) for v in means[id(c)].tolist()]
                                           for j, c in enumerate(cells)},
                      accuracy={})
        for name, make in interventions.items():
            for c in cells:
                c.decay_override = make(c)
            result["accuracy"][name] = accuracy(model, args.kind, meta, args, device)
        for c in cells:
            c.decay_override = None
        out = OUTDIR / f"{Path(path).stem}.json"
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"DONE {Path(path).stem} " + " ".join(f"{k}={v:.4f}" for k, v in result["accuracy"].items()), flush=True)


if __name__ == "__main__":
    main()
