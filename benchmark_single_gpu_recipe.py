"""Compare sequential and two-worker real-task timing on one GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from artifacts import atomic_write_json
from benchmark_kaggle_recipe import numerical_qualification, model_qualification


def project(summary, epochs):
    durations = sorted(summary['optimizer_step_seconds'])
    mean = statistics.mean(durations)
    conservative = max(mean * 1.2, durations[math.ceil(.95 * len(durations)) - 1])
    steps = math.ceil(summary['settings']['train_examples'] / summary['settings']['effective_batch']) * epochs
    overhead = epochs * (summary['validation_seconds'] + summary['checkpoint_seconds'])
    return dict(optimizer_steps=steps, mean_step_seconds=mean,
                estimated_seconds=mean * steps + overhead + summary['setup_seconds'],
                conservative_seconds=conservative * steps + epochs * summary['validation_seconds'] * 1.2
                + epochs * summary['checkpoint_seconds'] + 2 * summary['setup_seconds'] + 900)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['mqar', 'overwrite', 'both'], default='both')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=64)
    parser.add_argument('--cpu-smoke', action='store_true', help='Tiny plumbing check; not a GPU benchmark')
    args = parser.parse_args()
    if args.steps < 50 and not args.cpu_smoke or not 1 <= args.epochs <= 64:
        parser.error('Use at least 50 timing steps and 1..64 projection epochs')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='running', torch=torch.__version__, jobs={}, cpu_smoke=args.cpu_smoke,
                  source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ['recipe_model.py', 'recipe_data.py', 'recipe_evaluation.py',
                                              'experimental_chunked_recurrence.py', 'run_published_recipe.py']},
                  interpretation='Timing only; no learned accuracy qualification. Compare sequential and concurrent workers on device 0.')
    try:
        if not args.cpu_smoke and not torch.cuda.is_available():
            raise RuntimeError('CUDA PyTorch/GPU unavailable. Install a compatible CUDA-enabled PyTorch build.')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
        report['gpu'] = None if args.cpu_smoke else torch.cuda.get_device_name(0)
        proof = numerical_qualification(device, 16)
        proof['model_trajectory'] = model_qualification(device, 16)
        atomic_write_json(args.output / 'numerical_qualification.json', proof)
        if not args.cpu_smoke:
            torch.cuda.empty_cache()
        families = ['mqar', 'overwrite'] if args.family == 'both' else [args.family]
        for family in families:
            def measure(variant, mode):
                folder = args.output / family / mode / variant
                folder.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, 'run_published_recipe.py', '--worker', '--family', family,
                           '--variant', variant, '--mode', 'timing', '--backend', 'chunked',
                           '--output', str(folder), '--steps', str(args.steps), '--epochs', '64',
                           '--microbatch', '32', '--effective-batch', '512', '--validation-examples', '512']
                if args.cpu_smoke:
                    command += ['--cpu-smoke']
                print(f'Starting {family}/{mode}/{variant} on device 0', flush=True)
                with (folder.parent / f'{variant}.log').open('w', encoding='utf-8') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=39630, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1'))
                summary = json.loads((folder / 'summary.json').read_text())
                if summary['status'] != 'complete' or summary['accepted_optimizer_steps'] != args.steps:
                    raise RuntimeError('Incomplete timing worker')
                return summary, json.loads((folder / 'data_manifests.json').read_text())
            for mode in ['sequential', 'parallel']:
                tick = time.monotonic()
                variants = ['scalar', 'channel-wise']
                if mode == 'parallel':
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        futures = [pool.submit(measure, variant, mode) for variant in variants]
                        results = [future.result() for future in futures]
                else:
                    results = [measure(variant, mode) for variant in variants]
                elapsed = time.monotonic() - tick
                if results[0][1] != results[1][1]:
                    raise RuntimeError('Paired task data differ')
                for variant, (summary, _) in zip(variants, results):
                    report['jobs'][f'{family}/{mode}/{variant}'] = project(summary, args.epochs)
                projections = [report['jobs'][f'{family}/{mode}/{v}'] for v in variants]
                report.setdefault('comparisons', {}).setdefault(family, {})[mode] = dict(
                    benchmark_wall_seconds=elapsed, end_to_end_optimizer_steps_per_second=2 * args.steps / elapsed,
                    estimated_pair_training_seconds=(sum if mode == 'sequential' else max)(
                        j['estimated_seconds'] for j in projections),
                    conservative_pair_training_seconds=(sum if mode == 'sequential' else max)(
                        j['conservative_seconds'] for j in projections))
                atomic_write_json(args.output / 'benchmark_report.json', report)
                print(json.dumps(report['comparisons'][family][mode]), flush=True)
            comparison = report['comparisons'][family]
            comparison['recommended_mode_by_projected_pair_time'] = min(
                ['sequential', 'parallel'], key=lambda m: comparison[m]['estimated_pair_training_seconds'])
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['benchmark_wall_seconds'] = time.monotonic() - started
        atomic_write_json(args.output / 'benchmark_report.json', report)


if __name__ == '__main__':
    main()
