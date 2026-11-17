"""Layout-controlled confirmation of the block-first overwrite shortcut.

Discovery (research_notes/write_gate_mechanism_review_2026-11-17.md) found that trained overwrite models
fail almost only when every write of the queried key precedes every unrelated write ("block-first").
This script tests that on fresh data with layouts fixed in equal numbers, using the unchanged v4
generator conditioned on write order (rejection sampling), so examples differ from training only in
layout. Hypotheses and decision rules are written to protocol.json before any model is evaluated.

No training. Data seed offset 1300000 is new; confirmation offset 900000 stays untouched.
Pattern notation: T = a write of the queried key, U = an unrelated write, in memory order.
"""
import argparse
import csv
import gzip
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch

from analyze_write_gate import Positions, classify, load, select_joint
from artifacts import atomic_write_json
from config import RetrievalConfig
from confirm_recipe_study import select_runs
from recipe_data import RecipeData, build_data, manifest, split_overlap
from retrieval import RetrievalDataset

SEED_OFFSET = 1300000
SHORTCUT_GAP = .20       # block-first accuracy at least 20 points below general non-block
SPECIAL_MATCH = .10      # discriminating layouts within 10 points of general non-block
RULES = ('latest', 'initial', 'second_occurrence', 'memory_slot_1', 'opening_run_second', 'final_run_second')


def special_patterns(rewrites):
    n = rewrites + 1
    patterns = {'block_first': 'T' * n + 'UUU',
                'block_after_one_unrelated': 'U' + 'T' * n + 'UU',
                'run_broken_before_latest': 'T' * rewrites + 'U' + 'T' + 'UU'}
    if rewrites >= 2:
        patterns['initial_then_run'] = 'TU' + 'T' * rewrites + 'UU'
    return patterns


def cells():
    """(name, keys, distance, rewrites, layouts). Main grid plus distance controls."""
    out = []
    for rewrites in (1, 2, 3, 4):
        out.append((f'main_r{rewrites}', 4, 8, rewrites, [*special_patterns(rewrites), 'general_nonblock']))
    for distance in (12, 16):
        for rewrites in (1, 3):
            out.append((f'distance{distance}_r{rewrites}', 4, distance, rewrites, ['block_first', 'general_nonblock']))
    return out


def pattern_of(record):
    targets = set(record.metadata['write_positions'])
    return ''.join('T' if p in targets else 'U' for p in record.metadata['all_write_positions'])


def layout_of(pattern, rewrites):
    for name, special in special_patterns(rewrites).items():
        if pattern == special:
            return name
    return 'general_nonblock'


def sample_layouts(seed, cell_index, keys, distance, rewrites, layouts, examples, max_draws=1 << 19):
    cfg = RetrievalConfig(vocabulary_size=8192, sequence_lengths=(65,), key_counts=(keys,), query_distances=(distance,),
                          overwrite_counts=(rewrites,), generator_version='mqar-overwrite-v4')
    dataset_seed = seed + SEED_OFFSET + 1000 * cell_index
    dataset = RetrievalDataset(task='overwrite', examples=max_draws, config=cfg, seed=dataset_seed, split='held_out')
    chosen = {name: [] for name in layouts}
    draws = 0
    while any(len(v) < examples for v in chosen.values()):
        if draws >= max_draws:
            raise RuntimeError(f'Could not fill layouts {[(k, len(v)) for k, v in chosen.items()]}')
        record = dataset.example(draws)
        draws += 1
        name = layout_of(pattern_of(record), rewrites)
        if name in chosen and len(chosen[name]) < examples:
            chosen[name].append(record)
    return chosen, dict(dataset_seed=dataset_seed, draws=draws, manifest=dataset.manifest())


def to_cell(records, rewrites, identity):
    tokens = torch.stack([r.tokens for r in records])
    labels = torch.stack([r.labels for r in records])
    kinds = torch.full_like(labels, -1)
    stale = torch.full_like(labels, -1)
    erase = torch.zeros_like(tokens, dtype=torch.bool)
    for i, record in enumerate(records):
        for answer in record.answers:
            pos = answer.label_position
            kinds[i, pos] = 1 if answer.kind == 'latest' else 2
            if answer.kind == 'latest':
                stale[i, pos] = answer.stale_values[-1]
            if pos + 1 < tokens.shape[1]:
                erase[i, pos + 1] = True
    data = RecipeData(tokens, labels, kinds, stale, erase, manifest(tokens, labels, **identity))
    answer_free = tokens.clone()
    answer_free[erase] = 1
    return dict(case=SimpleNamespace(data=data), tokens=answer_free, positions=Positions(records),
                rewrites=rewrites, records=records)


def rule_answers(record):
    """Answer each candidate rule gives for the latest-target query."""
    values = record.metadata['write_values']
    pattern = pattern_of(record)
    memory_values = [record.tokens[p].item() if p < len(record.tokens) else None
                     for p in record.metadata['all_write_positions']]
    latest = values[-1]
    opening = pattern.startswith('T' * len(values))  # block-first: unbroken run from slot 0 through latest
    run = len(pattern[:pattern.rindex('T') + 1]) - len(pattern[:pattern.rindex('T') + 1].rstrip('T'))
    return dict(latest=latest, initial=values[0], second_occurrence=values[1],
                memory_slot_1=memory_values[1],
                opening_run_second=values[1] if opening else latest,
                final_run_second=values[len(values) - run + 1] if run >= 2 else latest)


def hypotheses():
    return {
        'H1_shortcut': 'In main_r3, a model shows the shortcut if block_first latest accuracy is at least 20 points '
                       'below general_nonblock. Confirmed if a majority of evaluated models show it (10 of 18 with joint models).',
        'H2_error_identity': 'Pooled over models and main_r2..r4, at least 80% of block_first errors return the '
                             'first replacement.',
        'H3_trained_layout_sanity': 'In main_r1 (training distribution), block_first latest accuracy is at least 95% '
                                    'in all but at most one evaluated model.',
        'H4_rule_specificity': 'Among models showing the shortcut in main_r3, block_after_one_unrelated, '
                               'run_broken_before_latest and initial_then_run are each within 10 points of '
                               'general_nonblock in a majority of those models (the failure needs the unbroken run '
                               'from slot 0).',
        'H5_gap_control': 'Open question, either result is reported: among models showing the shortcut, does the '
                          'block_first deficit (>=20 points) persist at query distance 12 and 16, where filler '
                          'separates memory from query? Persisting implicates the write run; vanishing implicates '
                          'memory-query adjacency or absolute position.',
    }


def evaluate(model, cell, device, batch):
    mask = cell['case'].data.labels != -100
    predictions = []
    with torch.no_grad():
        for start in range(0, len(mask), batch):
            predictions.append(model.scored_logits(cell['tokens'][start:start + batch].to(device),
                                                   mask[start:start + batch].to(device)).argmax(-1).cpu())
    predictions = torch.cat(predictions)
    outcomes, unrelated = classify(predictions, cell)
    per_row = mask.sum(1).tolist()
    latest_answers = []
    for i, row in enumerate(torch.split(predictions, per_row)):
        lookup = dict(zip(mask[i].nonzero().flatten().tolist(), row.tolist()))
        latest_answers.append(lookup[cell['positions'].latest_label[i]])
    return outcomes, unrelated, latest_answers


def verdicts(summary, models):
    def acc(model, cell, layout):
        return summary[model, cell, layout]['latest_value_accuracy']
    shortcut = [m for m in models if acc(m, 'main_r3', 'block_first') <= acc(m, 'main_r3', 'general_nonblock') - SHORTCUT_GAP]
    errors = Counter()
    for m in models:
        for r in (2, 3, 4):
            errors.update({k: v for k, v in summary[m, f'main_r{r}', 'block_first']['outcome_counts'].items() if k != 'correct'})
    h2 = errors['first_rewrite'] / max(1, sum(errors.values()))
    sane = [m for m in models if acc(m, 'main_r1', 'block_first') >= .95]
    h4 = {}
    for layout in ('block_after_one_unrelated', 'run_broken_before_latest', 'initial_then_run'):
        close = [m for m in shortcut if abs(acc(m, 'main_r3', layout) - acc(m, 'main_r3', 'general_nonblock')) <= SPECIAL_MATCH]
        h4[layout] = dict(models_within_10_points=len(close), of=len(shortcut))
    h5 = {}
    for d in (12, 16):
        persists = [m for m in shortcut if acc(m, f'distance{d}_r3', 'block_first')
                    <= acc(m, f'distance{d}_r3', 'general_nonblock') - SHORTCUT_GAP]
        h5[f'distance_{d}'] = dict(deficit_persists=len(persists), of=len(shortcut))
    n = len(models)
    return {
        'H1_shortcut': dict(models_showing=len(shortcut), of=n, confirmed=len(shortcut) > n / 2,
                            models=['/'.join(map(str, m)) for m in shortcut]),
        'H2_error_identity': dict(first_rewrite_share=h2, error_counts=dict(errors), confirmed=h2 >= .8),
        'H3_trained_layout_sanity': dict(models_at_least_95=len(sane), of=n, confirmed=len(sane) >= n - 1),
        'H4_rule_specificity': dict(by_layout=h4, confirmed=bool(shortcut) and all(
            v['models_within_10_points'] > v['of'] / 2 for v in h4.values())),
        'H5_gap_control': h5,
    }


def write_report(output, verdict, summary, models, rule_rows, wall):
    lines = ['# Block-first layout confirmation', '',
             f'Models: {len(models)}. Wall time {wall:.0f} s. Fresh data, seed offset {SEED_OFFSET}. '
             'Hypotheses were frozen in protocol.json before evaluation.', '', '## Verdicts', '']
    for key, value in verdict.items():
        lines.append(f'- **{key}**: `{json.dumps(value)}`')
    lines += ['', '## Latest-value accuracy by layout (mean over models)', '',
              '| Cell | Layout | Latest | Unrelated | First-replacement answers |', '|---|---|---:|---:|---:|']
    for (cell, layout) in sorted({(c, l) for _, c, l in summary}):
        rows = [summary[m, cell, layout] for m in models]
        lines.append(f"| {cell} | {layout} | {mean(r['latest_value_accuracy'] for r in rows):.1%} | "
                     f"{mean(r['unrelated_retention'] for r in rows):.1%} | {mean(r['first_rewrite_error_rate'] for r in rows):.1%} |")
    lines += ['', '## Per model, main_r3: block_first / general_nonblock / block_after_one / broken / initial_then_run', '',
              '| Model | block_first | general | after_one | broken | initial_then_run |', '|---|---:|---:|---:|---:|---:|']
    for m in models:
        vals = [summary[m, 'main_r3', l]['latest_value_accuracy'] for l in
                ('block_first', 'general_nonblock', 'block_after_one_unrelated', 'run_broken_before_latest', 'initial_then_run')]
        lines.append(f"| {'/'.join(map(str, m))} | " + ' | '.join(f'{v:.1%}' for v in vals) + ' |')
    lines += ['', '## Rule agreement on main_r2..r4, all layouts (share of latest-target answers equal to each rule)', '',
              '| Model | ' + ' | '.join(RULES) + ' |', '|---|' + '---:|' * len(RULES)]
    for row in rule_rows:
        lines.append(f"| {row['model']} | " + ' | '.join(f"{row[r]:.1%}" for r in RULES) + ' |')
    lines += ['', 'Interpretation limits: discovery-confirmation on synthetic data for 1.6M-parameter models; '
              'block_first is also the layout with the longest leading filler, so absolute position is only '
              'partly separated by the distance controls.']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/calibrated_recipe_study/matched-decay-v1'))
    parser.add_argument('--joint-root', type=Path)
    parser.add_argument('--joint-run-id', nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--initializations', nargs='+', default=['default', 'calibrated'])
    parser.add_argument('--variants', nargs='+', default=['scalar', 'channel-wise'])
    parser.add_argument('--examples', type=int, default=256, help='Examples per layout per cell')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--skip-overlap-audit', action='store_true', help='Smoke tests only')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = select_runs(args.root, ['overwrite'], args.seeds, [], 'explore')
    selected = {k: v for k, v in selected.items() if k[2] in args.initializations and k[3] in args.variants}
    if args.joint_root:
        joint = select_joint(args.joint_root, args.seeds, args.joint_run_id)
        selected.update({k: v for k, v in joint.items() if k[3] in args.variants})
    models = sorted((k[2], k[3], k[1]) for k in selected)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    data, data_audit = {}, []
    for seed in args.seeds:
        train = None if args.skip_overlap_audit else build_data('overwrite', 100000, seed + 100000, 'train', 8192, 64)
        for index, (name, keys, distance, rewrites, layouts) in enumerate(cells()):
            chosen, info = sample_layouts(seed, index, keys, distance, rewrites, layouts, args.examples)
            for layout, records in chosen.items():
                identity = dict(seed=info['dataset_seed'], split='layout_confirmation', cell=name, layout=layout,
                                keys=keys, distance=distance, rewrites=rewrites)
                cell = to_cell(records, rewrites, identity)
                overlap = None if train is None else split_overlap(train, cell['case'].data)
                if overlap:
                    raise ValueError(f'Confirmation data overlap training data: {name}/{layout}')
                data[seed, name, layout] = cell
                data_audit.append(dict(seed=seed, cell=name, layout=layout, draws=info['draws'], train_overlap=overlap,
                                       sha256=cell['case'].data.identity['sha256'],
                                       patterns=dict(Counter(pattern_of(r) for r in records))))
        del train
        print(f'Data for seed {seed} ready ({time.monotonic() - started:.0f}s)', flush=True)

    atomic_write_json(args.output / 'protocol.json', dict(
        purpose=__doc__, hypotheses=hypotheses(), thresholds=dict(shortcut_gap=SHORTCUT_GAP, special_match=SPECIAL_MATCH),
        seed_offset=SEED_OFFSET, examples_per_layout=args.examples, cells=cells(), device=str(device), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None, conditioning='answer_free',
        overlap_audit=not args.skip_overlap_audit, data=data_audit,
        checkpoints={'/'.join(map(str, k)): str(v[0].parent) for k, v in selected.items()}))

    summary, prediction_rows, rule_counts = {}, [], {}
    for key, (path, _) in sorted(selected.items()):
        _, seed, init, variant = key
        model = load(path, variant, seed).to(device)
        tag = (init, variant, seed)
        for (s, name, layout), cell in data.items():
            if s != seed:
                continue
            outcomes, unrelated, answers = evaluate(model, cell, device, args.batch)
            counts = Counter(outcomes)
            n = len(outcomes)
            summary[tag, name, layout] = dict(latest_value_accuracy=counts['correct'] / n,
                                              unrelated_retention=sum(unrelated) / n,
                                              first_rewrite_error_rate=counts['first_rewrite'] / n,
                                              outcome_counts=dict(counts), examples=n)
            for i, (record, outcome, answer) in enumerate(zip(cell['records'], outcomes, answers)):
                rules = rule_answers(record)
                if name in ('main_r2', 'main_r3', 'main_r4'):
                    tally = rule_counts.setdefault(tag, Counter())
                    tally['n'] += 1
                    tally.update(r for r in RULES if rules[r] == answer)
                prediction_rows.append(dict(initialization=init, variant=variant, seed=seed, cell=name, layout=layout,
                                            example=i, pattern=pattern_of(record), outcome=outcome, answer=answer,
                                            unrelated_correct=int(unrelated[i]), **{f'rule_{r}': rules[r] for r in RULES}))
        del model
        print(f'{init}/{variant}/{seed}: done ({time.monotonic() - started:.0f}s)', flush=True)

    verdict = verdicts(summary, models)
    rule_rows = [dict(model='/'.join(map(str, m)), **{r: rule_counts[m][r] / rule_counts[m]['n'] for r in RULES}) for m in models]
    wall = time.monotonic() - started
    atomic_write_json(args.output / 'results.json', dict(
        status='complete', wall_seconds=wall, verdicts=verdict, rule_agreement=rule_rows,
        cells=[dict(initialization=m[0], variant=m[1], seed=m[2], cell=c, layout=l, **v) for (m, c, l), v in summary.items()]))
    with (args.output / 'cell_results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['initialization', 'variant', 'seed', 'cell', 'layout', 'latest_value_accuracy',
                         'unrelated_retention', 'first_rewrite_error_rate', 'examples'])
        for (m, c, l), v in sorted(summary.items()):
            writer.writerow([*m, c, l, v['latest_value_accuracy'], v['unrelated_retention'],
                             v['first_rewrite_error_rate'], v['examples']])
    with gzip.open(args.output / 'predictions.csv.gz', 'wt', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    write_report(args.output, verdict, summary, models, rule_rows, wall)
    print(f'Complete in {wall:.0f}s. Verdicts:', flush=True)
    for k, v in verdict.items():
        print(' ', k, json.dumps(v), flush=True)


if __name__ == '__main__':
    main()
