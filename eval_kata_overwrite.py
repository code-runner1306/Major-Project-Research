"""Evaluate kata-overwrite-v1 models on write-count shifts (pre-registered predictions K1-K4).

Training wrote every key exactly twice. Evaluation writes one target key once, three or four times (others twice),
or every key once or three times. For each target-key query the returned value is attributed to the write it came
from (first, second, third, fourth write of that key), another key's value, or anything else.
Predictions are taken from run_kata_overwrite_study.PREDICTIONS and copied into protocol.json before evaluating.
Fresh data at seed offset 1700000, checked for exact overlap with each seed's training set. No training.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from artifacts import atomic_write_json
from kata_overwrite_data import EVAL_CELLS, build_kata_data, make_eval_cell, oracle_audit
from recipe_data import split_overlap
from recipe_model import RecipeLM
from run_kata_overwrite_study import PREDICTIONS, PROTOCOL, identity

SEED_OFFSET = 1700000
SHORTCUT_GAP = .20


def select(root, seeds):
    runs = {}
    for path in root.rglob('summary.json'):
        summary = json.loads(path.read_text())
        cfg = summary.get('settings', {})
        if cfg.get('protocol') != PROTOCOL or cfg.get('seed') not in seeds:
            continue
        if (summary.get('status') != 'complete' or summary.get('accepted_optimizer_steps') != 3136
                or cfg.get('source_hashes') != identity() or not (path.parent / 'latest_checkpoint.pt').is_file()):
            raise ValueError(f'Incomplete or incompatible run: {path}')
        key = f"{cfg['variant']}/{cfg['seed']}"
        if key in runs:
            raise ValueError(f'Duplicate run for {key}')
        runs[key] = (path.parent / 'latest_checkpoint.pt', cfg['variant'], cfg['seed'])
    if not runs:
        raise ValueError(f'No complete {PROTOCOL} runs under {root}')
    return runs


def attribute(answer, query, metas_values, vocabulary):
    writes = query['writes']
    if answer == writes[-1]:
        return 'correct'
    if answer in writes:
        return f'write_{writes.index(answer) + 1}'
    if answer in metas_values:
        return 'other_key_value'
    return 'other'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/kata_overwrite_study/kata-overwrite-v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--examples', type=int, default=512)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--skip-overlap-audit', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    runs = select(args.root, args.seeds)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cells, audits = {}, []
    for seed in sorted({r[2] for r in runs.values()}):
        train = None if args.skip_overlap_audit else build_kata_data(100000, seed + 100000, 'train')
        for index, name in enumerate(EVAL_CELLS):
            data, metas = make_eval_cell(name, args.examples, seed + SEED_OFFSET + 1000 * index)
            oracle = oracle_audit(data)
            overlap = None if train is None else split_overlap(train, data)
            if not oracle['passed'] or overlap:
                raise ValueError(f'Evaluation data audit failed: seed {seed} {name}')
            cells[seed, name] = (data, metas)
            audits.append(dict(seed=seed, cell=name, train_overlap=overlap, oracle=oracle, sha256=data.identity['sha256']))
        del train
    atomic_write_json(args.output / 'protocol.json', dict(purpose=__doc__, predictions=PREDICTIONS, seed_offset=SEED_OFFSET,
        shortcut_gap=SHORTCUT_GAP, cells={k: list(v) for k, v in EVAL_CELLS.items()}, examples=args.examples,
        device=str(device), runs={k: str(v[0]) for k, v in runs.items()}, data=audits))
    print(f'Data ready ({time.monotonic() - started:.0f}s)', flush=True)

    results, rows = {}, []
    for key, (path, variant, seed) in sorted(runs.items()):
        model = RecipeLM(variant, 8192, 64, 128, 'chunked', seed)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
        model = model.to(device).eval()
        for name in EVAL_CELLS:
            data, metas = cells[seed, name]
            mask = data.labels != -100
            predictions = []
            with torch.no_grad():
                for start in range(0, len(data.tokens), args.batch):
                    predictions.append(model.scored_logits(data.tokens[start:start + args.batch].to(device),
                                                           mask[start:start + args.batch].to(device)).argmax(-1).cpu())
            predictions = torch.cat(predictions)
            target, others, offset = Counter(), Counter(), 0
            for i, meta in enumerate(metas):
                positions = mask[i].nonzero().flatten().tolist()
                answers = dict(zip(positions, predictions[offset:offset + len(positions)].tolist()))
                offset += len(positions)
                all_values = {v for q in meta for v in q['writes']}
                for q in meta:
                    outcome = attribute(answers[q['position']], q, all_values - set(q['writes']), 8192)
                    (target if q['key'] == 0 else others)[outcome] += 1
            n_target, n_other = sum(target.values()), sum(others.values())
            results[key, name] = dict(target_accuracy=target['correct'] / n_target,
                                      other_accuracy=others['correct'] / n_other if n_other else None,
                                      target_outcomes=dict(target), other_outcomes=dict(others))
            rows.append(dict(model=key, variant=variant, seed=seed, cell=name, target_accuracy=results[key, name]['target_accuracy'],
                             other_accuracy=results[key, name]['other_accuracy'],
                             target_second_write=target['write_2'], target_errors=n_target - target['correct']))
        print(f'{key}: done ({time.monotonic() - started:.0f}s)', flush=True)

    models = sorted(runs)
    acc = lambda m, c: results[m, c]['target_accuracy']
    shortcut = [m for m in models if acc(m, 'target_three') <= acc(m, 'in_distribution') - SHORTCUT_GAP]
    errors = Counter()
    for m in models:
        for c in ('target_three', 'target_four'):
            errors.update({k: v for k, v in results[m, c]['target_outcomes'].items() if k != 'correct'})
    second = errors['write_2'] / max(1, sum(errors.values()))
    verdicts = {
        'K1_learned': dict(models_at_least_95=sum(acc(m, 'in_distribution') >= .95 for m in models), of=len(models)),
        'K2_rewrite_shortcut': dict(models_showing=len(shortcut), of=len(models), models=shortcut,
                                    confirmed=len(shortcut) >= 2),
        'K3_error_identity': dict(second_write_share=second, error_counts=dict(errors), confirmed=second >= .5),
        'K4_descriptive': {c: round(mean(acc(m, c) for m in models), 4) for c in ('target_once', 'all_once', 'all_three')},
    }
    wall = time.monotonic() - started
    atomic_write_json(args.output / 'results.json', dict(status='complete', wall_seconds=wall, verdicts=verdicts,
                      results=[dict(model=m, cell=c, **v) for (m, c), v in results.items()]))
    with (args.output / 'cell_results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# KATA-style overwrite evaluation', '', f'Models: {len(models)}. Wall time {wall:.0f} s. Seed offset '
             f'{SEED_OFFSET}. Predictions frozen before training (run_kata_overwrite_study.PREDICTIONS).', '', '## Verdicts', '']
    lines += [f'- **{k}**: `{json.dumps(v)}`' for k, v in verdicts.items()]
    lines += ['', '## Target-key accuracy (other keys in brackets)', '', '| Model | ' + ' | '.join(EVAL_CELLS) + ' |',
              '|---|' + '---:|' * len(EVAL_CELLS)]
    for m in models:
        cells_text = []
        for c in EVAL_CELLS:
            other = results[m, c]['other_accuracy']
            cells_text.append(f"{acc(m, c):.1%}" + (f" ({other:.1%})" if other is not None else ''))
        lines.append(f'| {m} | ' + ' | '.join(cells_text) + ' |')
    lines += ['', '## Target-key error attribution (all models)', '']
    for c in EVAL_CELLS:
        pooled = Counter()
        for m in models:
            pooled.update({k: v for k, v in results[m, c]['target_outcomes'].items() if k != 'correct'})
        lines.append(f'- {c}: {dict(pooled)}')
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(verdicts, indent=1), flush=True)


if __name__ == '__main__':
    main()
