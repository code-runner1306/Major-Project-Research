"""Four-key learnability comparison; development evidence, not a causal verdict."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from artifacts import atomic_write_json
from config import RetrievalConfig, TrainingBudget, production_profile, smoke_profile
from controls import construct_model
from energy import EnergyProvider
from fixed_learnability import compact
from model import CausalSelfAttention, GatedDeltaNetLM
from orchestration import _resolved_backbone, recurrence_options_for
from retrieval import RetrievalDataset, evaluate_retrieval, evaluate_retrieval_baselines, retrieval_dataloader
from train import SeedBundle, seed_everything, train_fixed_budget


def metrics_summary(value):
    return {**compact(value), **{key: value.get(key) for key in
        ('latest_value_accuracy', 'unrelated_association_retention', 'stale_value_error_rate')}}


class AttentionControl(GatedDeltaNetLM):
    """Same LM backbone, full causal attention in every layer, learned positions."""
    def __init__(self, backbone, vocabulary, options, seed, context):
        super().__init__(backbone, 'scalar', vocabulary, options, study_init_seed=seed)
        for block in self.blocks:
            block.attn = CausalSelfAttention(backbone.d_model, backbone.n_head, backbone.d_head)
        self.positions = nn.Embedding(context, backbone.d_model)
        self.initialize_study_weights(seed)

    def forward(self, tokens, targets=None):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        hidden = self.tok_embeddings(tokens) + self.positions(positions)[None]
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.output(self.norm(hidden))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss


class Indexed(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return (*self.data[index], index)


class TrackingLoader:
    def __init__(self, data, batch, seed):
        self.loader = DataLoader(Indexed(data), batch_size=batch, shuffle=True,
                                generator=torch.Generator().manual_seed(seed), num_workers=0)
        self.first = []
        self.seen = set()
        self.recent = deque(maxlen=512)
    def __iter__(self):
        for tokens, labels, indices in self.loader:
            for index in indices.tolist():
                if index not in self.seen:
                    self.first.append(index)
                    self.seen.add(index)
                self.recent.append(index)
            yield tokens, labels


def datasets(family, seed, cpu=False):
    config = RetrievalConfig(vocabulary_size=128 if cpu else 8192,
        train_examples=32 if cpu else 10000, validation_examples=16 if cpu else 128,
        held_out_examples=32 if cpu else 512, sequence_lengths=(64 if cpu else 128,),
        key_counts=(4,), query_distances=(8,), overwrite_counts=(1,), generator_version='mqar-overwrite-v4')
    train = RetrievalDataset(task=family, examples=config.train_examples, config=config, seed=seed + 100000, split='train')
    dev = RetrievalDataset(task=family, examples=config.validation_examples, config=config, seed=seed + 200000, split='validation')
    test = RetrievalDataset(task=family, examples=config.held_out_examples, config=config, seed=seed + 300000, split='held_out')
    return config, train, dev, test


def audit(family, seed, cpu=False):
    config, train, dev, test = datasets(family, seed, cpu)
    # Audit a meaningful sample even for implementation smoke checks.
    output = {}
    for split, data in (('development', dev), ('confirmation', test)):
        sample = RetrievalDataset(task=family, examples=max(128, len(data)), config=config, seed=data.seed, split=data.split)
        result = evaluate_retrieval_baselines(sample, frequency_dataset=train)
        output[split] = result
    output['passed'] = all(r['status'] == 'complete' and r['oracle_audit']['pass']
        and r['counterfactual_query_key_audit']['pass'] and r['shortcut_criterion']['all_cells_pass']
        and r['completeness']['incomplete_checks'] == 0 for r in output.values())
    return output


def worker(args):
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    config, train, dev, test = datasets(args.family, args.seed, args.cpu_smoke)
    profile = smoke_profile(args.output) if args.cpu_smoke else production_profile(args.output)
    scale = 'tiny' if args.cpu_smoke else '10M'
    profile = replace(profile, budget=TrainingBudget(args.steps, 4, 8, config.sequence_lengths[0],
        16, args.eval_every, status='study_locked_budget'), learning_rate=args.learning_rate,
        retrieval=config, recurrence_backend='reference', state_accumulator_dtype='float32')
    seed_everything(args.seed)
    backbone = _resolved_backbone(profile, args.family, scale)
    options = recurrence_options_for(profile)
    if args.architecture == 'gdn':
        model, _, _ = construct_model(backbone, 'scalar', config.vocabulary_size,
                                     recurrence_options=options, study_init_seed=args.seed)
    else:
        model = AttentionControl(backbone, config.vocabulary_size, options, args.seed, config.sequence_lengths[0])
    device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    model.to(device)
    loader = TrackingLoader(train, 8, args.seed + 100000)
    progress = []
    manifests = {'train': train.manifest(), 'development': dev.manifest(), 'confirmation': test.manifest()}
    atomic_write_json(args.output / 'settings.json', {'architecture': args.architecture,
        'family': args.family, 'seed': args.seed, 'steps': args.steps, 'learning_rate': args.learning_rate,
        'model_parameters': sum(p.numel() for p in model.parameters()), 'backbone': backbone.to_dict(),
        'manifests': manifests, 'attention_positions': 'learned absolute' if args.architecture == 'attention' else None})
    def measure(current, step):
        recent = list(dict.fromkeys(reversed(loader.recent)))[:128]
        row = {'step': step, 'distinct_train_examples_seen': len(loader.seen)}
        for name, data, indices in (('first_train', train, loader.first[:128]),
                                    ('recent_train', train, recent), ('development', dev, None)):
            for history in ('teacher_forced', 'answer_free'):
                row[f'{name}_{history}'] = metrics_summary(evaluate_retrieval(current, data, device,
                    example_indices=indices, history_conditioning=history))
        progress.append(row)
        atomic_write_json(args.output / 'progress.json', progress)
        print(json.dumps(row), flush=True)
        return {'development_answer_accuracy': row['development_answer_free']['exact_query_accuracy']}
    result = train_fixed_budget(model=model, train_dataloader=loader,
        validation_dataloader=retrieval_dataloader(dev, 8), profile=profile, device=device,
        run_dir=str(args.output), energy_provider=EnergyProvider.create(device),
        validation_metrics_fn=measure, progress_label=f'{args.family}/{args.architecture}')
    # Confirmation data is never used for model or checkpoint selection.
    confirmation = {history: metrics_summary(evaluate_retrieval(model, test, device, history_conditioning=history))
                    for history in ('teacher_forced', 'answer_free')}
    atomic_write_json(args.output / 'training_metrics.json', result)
    summary = {'status': 'complete', 'architecture': args.architecture, 'family': args.family,
        'steps': args.steps, 'last_development_measurement': progress[-1], 'confirmation': confirmation,
        'wall_seconds': time.monotonic() - start, 'seed': args.seed, 'manifests': manifests,
        'gradient_mean': statistics.mean(result['training']['unclipped_gradient_norms']),
        'gradient_max': max(result['training']['unclipped_gradient_norms']),
        'clipped_step_fraction': sum(x > profile.grad_clip for x in result['training']['unclipped_gradient_norms']) / args.steps,
        'amp_rejected_attempts': result['training']['rejected_optimizer_attempts'],
        'first_train_loss': result['training']['train_losses'][0] / 4,
        'last_train_loss': result['training']['train_losses'][-1] / 4,
        'interpretation': 'One seed and one LR; architecture and useful parameter counts differ. Diagnostic only.'}
    atomic_write_json(args.output / 'summary.json', summary)


def compare(root):
    report = {'purpose': 'four-key attention positive-control diagnostic', 'families': {}}
    for family in ('mqar', 'overwrite'):
        rows = {}
        for arch in ('gdn', 'attention'):
            path = root / family / arch / 'summary.json'
            rows[arch] = json.loads(path.read_text()) if path.exists() else {'status': 'incomplete'}
        if all(r['status'] == 'complete' for r in rows.values()):
            if rows['gdn']['manifests'] != rows['attention']['manifests']:
                report['families'][family] = {'runs': rows, 'finding': 'Invalid comparison: data identities differ.'}
                continue
            a = rows['attention']['confirmation']['answer_free']['exact_query_accuracy']
            g = rows['gdn']['confirmation']['answer_free']['exact_query_accuracy']
            if a >= .8 and g >= .8: finding = 'Both learn this setting; qualify the paper comparison next.'
            elif a >= .8 and g < .1: finding = 'Attention positive control succeeds; investigate GDN-specific optimization/implementation/capacity.'
            elif max(a, g) < .1: finding = 'Neither establishes learnability; shared setup or budget needs diagnosis.'
            else: finding = 'Intermediate outcome; inspect curves and per-query results before selecting a next run.'
        else: finding = 'Incomplete: do not infer an architecture result.'
        report['families'][family] = {'runs': rows, 'finding': finding}
    atomic_write_json(root / 'comparison_report.json', report)
    return report


def coordinator(args):
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.cpu_smoke and torch.cuda.device_count() < 2: raise RuntimeError('Two CUDA GPUs required')
    # Absolute ceiling includes audits, both task waves and final report writing.
    deadline = start + args.max_seconds - args.reserve_seconds
    status = {'status': 'running', 'jobs': {}, 'limit_seconds': args.max_seconds,
              'reserve_seconds': args.reserve_seconds, 'source_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()}
    code = 0
    old_alarm = None
    if os.name == 'posix':
        def expired(signum, frame): raise TimeoutError('Global time reserve reached')
        old_alarm = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.monotonic()))
    try:
        for family in ('mqar', 'overwrite'):
            result = audit(family, args.seed, args.cpu_smoke)
            atomic_write_json(args.output / f'{family}_audit.json', result)
            if not result['passed']: raise RuntimeError(f'{family} data audit failed; no training launched')
        for family in ('mqar', 'overwrite'):
            processes, logs = {}, []
            try:
                for gpu, arch in enumerate(('gdn', 'attention')):
                    if time.monotonic() >= deadline: raise TimeoutError('Time reserve reached')
                    log = (args.output / f'{family}_{arch}.log').open('w')
                    logs.append(log)
                    command = [sys.executable, __file__, '--worker', '--family', family, '--architecture', arch,
                        '--steps', str(args.steps), '--eval-every', str(args.eval_every), '--seed', str(args.seed),
                        '--learning-rate', str(args.learning_rate), '--output', str(args.output / family / arch)]
                    if args.cpu_smoke: command.append('--cpu-smoke')
                    processes[arch] = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)), start_new_session=True)
                    print(f'LAUNCHED {family}/{arch} on GPU {gpu}', flush=True)
                while any(p.poll() is None for p in processes.values()):
                    if time.monotonic() >= deadline: raise TimeoutError('Time reserve reached; saving partial evidence')
                    time.sleep(1)
                for arch, process in processes.items(): status['jobs'][f'{family}/{arch}'] = process.returncode
                atomic_write_json(args.output / 'session_status.json', status)
                if any(p.returncode for p in processes.values()): code = 1
            finally:
                for process in processes.values():
                    if process.poll() is None:
                        if os.name == 'posix': os.killpg(process.pid, signal.SIGTERM)
                        else: process.terminate()
                for process in processes.values():
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if os.name == 'posix': os.killpg(process.pid, signal.SIGKILL)
                        else: process.kill()
                        process.wait()
                for log in logs: log.close()
        status['status'] = 'complete' if code == 0 else 'incomplete'
    except (Exception, KeyboardInterrupt) as error:
        status.update(status='incomplete', error=str(error))
        code = 1
    finally:
        if old_alarm is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_alarm)
        status['wall_seconds'] = time.monotonic() - start
        atomic_write_json(args.output / 'session_status.json', status)
        print(json.dumps(compare(args.output), indent=2), flush=True)
        print(json.dumps(status, indent=2), flush=True)
        if args.push_results:
            from kaggle_session import export_evidence, remote_url, publish
            try:
                bundle = export_evidence(args.output)
                publish(bundle, remote_url('origin'), 'diagnostic/' + args.output.name)
            except Exception as error:
                print(f'Publishing failed; local results retained: {error}', file=sys.stderr)
                code = 1
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--eval-every', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--max-seconds', type=int, default=39600, help='11h total ceiling; leaves space within Kaggle 12h for setup')
    p.add_argument('--reserve-seconds', type=int, default=600)
    p.add_argument('--cpu-smoke', action='store_true')
    p.add_argument('--push-results', action='store_true', help='Push evidence-only branch using Git credentials or GH_TOKEN')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--family', choices=('mqar','overwrite'))
    p.add_argument('--architecture', choices=('gdn','attention'))
    args = p.parse_args()
    args.output = args.output.resolve()
    if args.steps < 2 or args.eval_every < 1 or not args.learning_rate > 0: p.error('Invalid training budget')
    if not 0 < args.reserve_seconds < args.max_seconds <= 39600: p.error('Total ceiling must be <=11h with positive reserve')
    if args.worker:
        if not args.family or not args.architecture: p.error('Worker requires family and architecture')
        worker(args)
        return 0
    return coordinator(args)


if __name__ == '__main__': raise SystemExit(main())
