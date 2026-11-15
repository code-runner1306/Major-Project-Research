"""Bounded V4 varied-data diagnostic; not a production research result."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import torch

from config import RetrievalConfig, TrainingBudget, production_profile
from controls import construct_model
from energy import EnergyProvider
from orchestration import _resolved_backbone, recurrence_options_for
from retrieval import RetrievalDataset, evaluate_retrieval, evaluate_retrieval_baselines, retrieval_dataloader
from train import SeedBundle, seed_everything, train_fixed_budget


def compact(metrics):
    keys = ('exact_query_accuracy', 'correct', 'total', 'example_accuracy',
            'first_query_accuracy', 'later_query_accuracy', 'evaluated_examples')
    return {key: metrics.get(key) for key in keys}


def _answer_free_curve(progress, split):
    curve = [{'step': row['step'],
              'exact_query_accuracy': row[f'{split}_answer_free']['exact_query_accuracy']}
             for row in progress]
    best = max(curve, key=lambda point: point['exact_query_accuracy']) if curve else None
    return {'points': curve,
            'final': curve[-1]['exact_query_accuracy'] if curve else None,
            'best': best['exact_query_accuracy'] if best else None,
            'best_step': best['step'] if best else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', required=True, choices=('mqar', 'overwrite'))
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-examples', type=int, default=10_000,
                        help='Number of deterministic generated training examples (128 reproduces the fixed-data test)')
    parser.add_argument('--train-eval-examples', type=int, default=128,
                        help='Number of training-set examples measured at each checkpoint')
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-smoke', action='store_true', help='Use a tiny CPU model for implementation checking')
    args = parser.parse_args()
    if args.steps < 2 or args.eval_every < 1 or args.train_examples < 8 or args.train_eval_examples < 1:
        parser.error('steps must be >= 2, train-examples >= 8 and evaluation settings positive')
    if not args.learning_rate > 0:
        parser.error('learning-rate must be positive')
    if not args.cpu_smoke and not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    seed = SeedBundle.from_research_seed(args.seed)
    # Keep the production 8192 vocabulary; simplify context, keys and distance.
    config = RetrievalConfig(vocabulary_size=8192, train_examples=args.train_examples, validation_examples=128,
        held_out_examples=128, sequence_lengths=(128,), key_counts=(2,), query_distances=(8,),
        overwrite_counts=(1,), generator_version='mqar-overwrite-v4')
    if args.cpu_smoke:
        config = replace(config, vocabulary_size=128, train_examples=8, validation_examples=8, held_out_examples=8,
                         sequence_lengths=(32,), query_distances=(8,))
    budget = TrainingBudget(max_optimizer_steps=args.steps, grad_accum_steps=4, batch_size=8,
        context_length=config.sequence_lengths[0], validation_steps=16,
        evaluation_interval=args.eval_every, status='study_locked_budget')
    if args.cpu_smoke:
        from config import smoke_profile
        profile = smoke_profile(args.output)
        scale = 'tiny'
    else:
        profile = production_profile(args.output)
        scale = '10M'
    profile = replace(profile, budget=budget, learning_rate=args.learning_rate, retrieval=config,
        recurrence_backend='reference', state_accumulator_dtype='float32')
    train = RetrievalDataset(task=args.family, examples=config.train_examples, config=config,
                             seed=seed.data_seed, split='train')
    held = RetrievalDataset(task=args.family, examples=config.held_out_examples, config=config,
                            seed=seed.evaluation_seed + 1, split='held_out')
    assert train.manifest()['hash'] != held.manifest()['hash']
    # Fixed random sample of the training set: with a shuffled training loader
    # the dataset-order prefix is no longer special, so measure a seeded sample
    # instead of the first examples in storage order.
    sample_rng = random.Random(seed.evaluation_seed)
    train_indices = tuple(sorted(sample_rng.sample(range(len(train)),
                                                   min(args.train_eval_examples, len(train)))))
    seed_everything(seed.model_seed)
    model, _, metadata = construct_model(_resolved_backbone(profile, args.family, scale), 'scalar',
        config.vocabulary_size, recurrence_options=recurrence_options_for(profile),
        gate_initialization='default', study_init_seed=seed.model_seed)
    device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    model.to(device)
    progress = []
    def measure(current_model, step):
        row = {'step': step}
        for name, data in (('train_sample', train), ('held_out', held)):
            for history in ('teacher_forced', 'answer_free'):
                row[f'{name}_{history}'] = compact(evaluate_retrieval(current_model, data, device,
                    history_conditioning=history, example_indices=train_indices if name == 'train_sample' else None))
        progress.append(row)
        (args.output / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        print('DIAGNOSTIC:', json.dumps(row), flush=True)
        return {'train_sample_answer_accuracy': row['train_sample_answer_free']['exact_query_accuracy'],
                'held_out_answer_accuracy': row['held_out_answer_free']['exact_query_accuracy']}
    # The frequency baseline sees only the prefix the model will actually see,
    # not future examples from a larger configured training set.
    seen_count = min(len(train), args.steps * budget.batch_size * budget.grad_accum_steps)
    seen_train = RetrievalDataset(task=args.family, examples=seen_count, config=config,
                                  seed=seed.data_seed, split='train')
    baseline = evaluate_retrieval_baselines(held, frequency_dataset=seen_train)
    (args.output / 'held_out_baselines.json').write_text(json.dumps(baseline, indent=2) + '\n')
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed.data_seed)
    result = train_fixed_budget(model=model,
        train_dataloader=retrieval_dataloader(train, budget.batch_size, shuffle=True,
                                              generator=shuffle_generator),
        validation_dataloader=retrieval_dataloader(train, budget.batch_size), profile=profile, device=device,
        run_dir=str(args.output), energy_provider=EnergyProvider.create(device),
        progress_label=f'fixed-{args.family}', validation_metrics_fn=measure)
    summary = {'status': 'complete', 'purpose': 'exploratory varied-data overfit and easy held-out learnability',
        'family': args.family, 'seed': args.seed, 'optimizer_steps': args.steps,
        'learning_rate': args.learning_rate, 'train_shuffle': True,
        'epochs_completed': (args.steps * budget.batch_size * budget.grad_accum_steps) / max(len(train), 1),
        'train_examples': len(train), 'held_out_examples': len(held),
        'distinct_train_examples_seen': seen_count,
        'train_accuracy_sample_examples': len(train_indices),
        'held_out_interpretation': 'development set monitored repeatedly; use fresh data for final confirmation',
        'sequence_length': config.sequence_lengths[0], 'key_count': 2, 'query_distance': config.query_distances[0],
        'vocabulary_size': config.vocabulary_size, 'generator_version': config.generator_version,
        'variant': 'scalar/default', 'parameter_count': metadata['parameter_count'],
        'train_manifest': train.manifest(), 'held_out_manifest': held.manifest(),
        'baseline_path': str(args.output / 'held_out_baselines.json'),
        'held_out_baselines': baseline['baselines'],
        'held_out_oracle_audit': baseline['oracle_audit'],
        'shortcut_audit_passed': baseline['shortcut_criterion']['all_cells_pass'],
        'last_measurement': progress[-1], 'all_measurements': progress,
        'held_out_curve': _answer_free_curve(progress, 'held_out'),
        'train_sample_curve': _answer_free_curve(progress, 'train_sample'),
        'training_wall_seconds': result['training']['wall_time_seconds'],
        'total_wall_seconds': time.monotonic() - start}
    (args.output / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('COMPLETE:', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
