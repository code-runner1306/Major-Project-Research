"""Replicate the KATA Table 2 Gated DeltaNet (dh=64) repeated-key overwrite row, then test it for the rewrite shortcut.

Protocol kata-replication-v1. Paper: Ghriss and Chakraborty, "Kernelized Linear Attention: Breaking the Capacity Wall
with Symmetric Cones", arXiv 2607.17419v1, Section 5.2, Table 2 and Appendix F.

Training as in Appendix F.3/F.4: batch 256, AdamW (beta 0.9/0.95), weight decay 0.1 except A_log and dt_bias, gradient
clipping 1.0, 32-epoch cosine schedule without warmup (stepped per epoch), FP32, and one run per learning rate in
{1e-3, 3e-3, 1e-2}; the paper reports the best learning rate per row. Training also stops early when the hardest
extrapolation slice (K=384) reaches 0.999 accuracy, as in F.4. Data and model: kata_overwrite_data.py, kata_model.py.

Three models are trained, one per learning rate, on seed 42; they run concurrently on one GPU by default.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from artifacts import atomic_write_json
from kata_model import KataLM, PAPER_PARAMETERS, parameter_count
from kata_overwrite_data import TRAIN_KEYS, generate, oracle_check, training_set

PROTOCOL = 'kata-replication-v1'
LEARNING_RATES = (1e-3, 3e-3, 1e-2)
PAPER_ROW = {'model': 'Gated DeltaNet, dh=64, 1,167,878 parameters', 'K=64': .998, 'K=128': .853, 'K=256': .384,
             'K=384': .217, 'average': .613}
PREDICTIONS = {
    'R1_replication': 'Descriptive comparison of the selected learning rate (best mean accuracy over K=64,128,256,384, '
                      'the paper\'s selection rule) with the paper row 0.998 / 0.853 / 0.384 / 0.217. We call the '
                      'replication adequate if K=64 accuracy is at least 0.95.',
    'R2_rewrite_shortcut': 'Selected model: accuracy with three write passes (K in 4..32) is at least 20 points below '
                           'two passes. Also reported for the other two learning rates.',
    'R3_error_identity': 'Pooled over the three learning rates, at least 50% of three-pass errors return the pass-2 '
                         '(second) write.',
    'R4_partial_third_pass': 'Descriptive: with a third pass rewriting half of the keys, accuracy on rewritten keys '
                             'versus keys left at their pass-2 value separates "second write" (fails rewritten keys) '
                             'from "value in the last block" (fails keys absent from it) and from latest-write recall.',
    'R5_single_pass': 'Descriptive: accuracy with one write pass (plain MQAR, never seen in training).',
    'evaluation': 'eval_kata_replication.py after all three runs finish; fresh evaluation seed 1800000.'}


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    smoke = args.cpu_smoke
    settings = dict(protocol=PROTOCOL, seed=args.seed, learning_rate=args.learning_rate, epochs=2 if smoke else 32,
                    batch=256, train_examples=400 if smoke else 100000, train_keys=list(TRAIN_KEYS), d_model=128, heads=2,
                    vocabulary=8192, optimizer='AdamW(beta1=0.9,beta2=0.95,weight_decay=0.1 except A_log,dt_bias)',
                    gradient_clipping=1.0, schedule='CosineAnnealingLR T_max=32 epochs, stepped per epoch, no warmup',
                    early_stop='stop when K=384 two-pass accuracy reaches 0.999', precision='float32',
                    paper_row=PAPER_ROW, torch=str(torch.__version__))
    atomic_write_json(args.output / 'settings.json', settings)
    train = training_set(settings['train_examples'], args.seed)
    development = {k: generate(64 if smoke else 256, k, 2, args.seed, 'validation')[:2] for k in TRAIN_KEYS}
    hardest = generate(4 if smoke else 64, 384, 2, args.seed, 'early_stop')[:2]
    audit = dict(train_oracle_failures=sum(oracle_check(t[:500], l[:500]) for t, l in train.values()),
                 development_oracle_failures=sum(oracle_check(t, l) for t, l in development.values()))
    if any(audit.values()):
        raise RuntimeError(f'Data audit failed: {audit}')
    atomic_write_json(args.output / 'data_audit.json', audit)

    model = KataLM(seed=args.seed).to(device)
    count = parameter_count(model)
    optimizer = model.optimizer(args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, eta_min=0.)
    generator = torch.Generator().manual_seed(args.seed + 1000)

    def accuracy(tokens, labels, batch=32):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for start in range(0, len(tokens), batch):
                t, l = tokens[start:start + batch].to(device), labels[start:start + batch].to(device)
                mask = l != -100
                correct += int((model.scored_logits(t, mask).argmax(-1) == l[mask]).sum())
                total += int(mask.sum())
        model.train()
        return correct / total

    curve, steps, stopped_early = [], 0, False
    model.train()
    for epoch in range(settings['epochs']):
        batches = [(k, idx) for k, (t, _) in train.items()
                   for idx in torch.randperm(len(t), generator=generator).split(settings['batch'])]
        order = torch.randperm(len(batches), generator=generator).tolist()
        losses = []
        for index in order:
            if time.monotonic() - started > args.max_seconds - 300:
                raise TimeoutError('Worker time budget reached')
            k, idx = batches[index]
            tokens, labels = train[k][0][idx].to(device), train[k][1][idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(tokens, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps += 1
            losses.append(loss.item())
        scheduler.step()
        row = dict(epoch=epoch + 1, steps=steps, learning_rate=optimizer.param_groups[0]['lr'],
                   train_loss=sum(losses) / len(losses),
                   development={f'K={k}': accuracy(*development[k]) for k in TRAIN_KEYS},
                   hardest_K384=accuracy(*hardest, batch=8), seconds=time.monotonic() - started)
        curve.append(row)
        atomic_write_json(args.output / 'curve.json', curve)
        torch.save(dict(model=model.state_dict(), settings=settings, epoch=epoch + 1, steps=steps), args.output / 'checkpoint.tmp')
        (args.output / 'checkpoint.tmp').replace(args.output / 'latest_checkpoint.pt')
        print(json.dumps(row), flush=True)
        if row['hardest_K384'] >= .999:
            stopped_early = True
            break
    atomic_write_json(args.output / 'summary.json', dict(
        status='complete', settings=settings, parameters=count, paper_parameters=PAPER_PARAMETERS,
        epochs_run=len(curve), steps=steps, stopped_early=stopped_early, final=curve[-1],
        wall_seconds=time.monotonic() - started, gpu=None if smoke else torch.cuda.get_device_name(0)))


def coordinator(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    if not args.cpu_smoke and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    atomic_write_json(args.output / 'study_protocol.json', dict(
        protocol=PROTOCOL, learning_rates=args.learning_rates, seed=args.seed, predictions=PREDICTIONS,
        paper_row=PAPER_ROW, concurrent=args.concurrent, cpu_smoke=args.cpu_smoke, torch=str(torch.__version__)))
    pending = list(args.learning_rates)
    running, status = [], dict(status='running', completed=[], failed=[])
    try:
        while pending or running:
            while pending and len(running) < args.concurrent:
                lr = pending.pop(0)
                folder = args.output / f'lr_{lr:g}'
                log = (args.output / f'lr_{lr:g}.log').open('w', encoding='utf-8')
                command = [sys.executable, __file__, '--worker', '--learning-rate', str(lr), '--seed', str(args.seed),
                           '--output', str(folder), '--max-seconds', str(int(args.max_seconds - (time.monotonic() - started)))]
                if args.cpu_smoke:
                    command.append('--cpu-smoke')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                running.append((lr, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env), log))
                print(f'Started learning rate {lr:g}', flush=True)
            for item in list(running):
                lr, process, log = item
                if process.poll() is not None:
                    log.close()
                    running.remove(item)
                    (status['completed'] if process.returncode == 0 else status['failed']).append(lr)
                    print(f'Learning rate {lr:g}: {"complete" if process.returncode == 0 else "FAILED, see log"}', flush=True)
            time.sleep(1)
        status['status'] = 'complete' if not status['failed'] else 'failed_or_incomplete'
    finally:
        for _, process, log in running:
            process.terminate()
            log.close()
        status['wall_seconds'] = time.monotonic() - started
        atomic_write_json(args.output / 'session_status.json', status)
    if status['failed']:
        raise RuntimeError(f'Runs failed: {status["failed"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rates', type=float, nargs='+', default=list(LEARNING_RATES))
    parser.add_argument('--concurrent', type=int, default=3, help='Runs sharing the GPU at once (default: all three)')
    parser.add_argument('--device', default='0')
    parser.add_argument('--max-seconds', type=int, default=39600)
    parser.add_argument('--cpu-smoke', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--learning-rate', type=float, help=argparse.SUPPRESS)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.plan_only:
        print(json.dumps(dict(protocol=PROTOCOL, models=len(args.learning_rates), learning_rates=args.learning_rates,
                              seed=args.seed, concurrent=args.concurrent, output=str(args.output)), indent=2))
        return
    if args.worker:
        worker(args)
    else:
        coordinator(args)


if __name__ == '__main__':
    main()
