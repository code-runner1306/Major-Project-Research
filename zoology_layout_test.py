"""Layout-stratified test of models trained on the standard Zoology MQAR generator.

Question: do models trained on data we did not design (pinned upstream Zoology MQAR, recipe_data.mqar, byte-equivalent
to Zoology random_non_queries=False) also hide position-based shortcuts behind near-perfect accuracy?

Zoology MQAR has fixed structure: the key-value memory always starts at position 0, keys always sit at even positions
and values at odd positions, and queries follow at power-law gaps (near ones much more likely). This script keeps
everything else as in Zoology (vocabulary halves, filler token 0, answer-free scoring at the query key) and changes
one layout property at a time:

  zoology_baseline      upstream generator, fresh seed (sanity check)
  shift_<s>             s filler tokens before the memory; odd s also flips key/value parity
  near / far            all queries in the first / last available slots
  memory_order / reverse_order   query order equal to / reversed from memory order (power-law gaps)
  shift_8_far           shifted memory and far queries together

Errors are attributed: the value of another memory pair, and specifically the "position-anchored" answer, meaning the
token sitting where the correct value would be in the unshifted Zoology layout. Hypotheses are written to protocol.json
before evaluation. No training. Seed offset 1600000 is new.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from artifacts import atomic_write_json
from confirm_recipe_study import select_runs
from recipe_data import build_data, mqar, split_overlap, RecipeData, manifest
from recipe_model import RecipeLM

SEED_OFFSET = 1600000
PAIRS, LENGTH, VOCAB = 4, 64, 8192
CELLS = [('zoology_baseline', 0, 'upstream'), ('shift_0', 0, 'power'), ('shift_1', 1, 'power'), ('shift_2', 2, 'power'),
         ('shift_4', 4, 'power'), ('shift_8', 8, 'power'), ('shift_16', 16, 'power'), ('near', 0, 'near'),
         ('far', 0, 'far'), ('memory_order', 0, 'memory_order'), ('reverse_order', 0, 'reverse_order'),
         ('shift_8_far', 8, 'far')]
SHORTCUT_GAP = .20


def hypotheses():
    return {
        'Z1_sanity': 'zoology_baseline accuracy >= 95% for every model (they are near-perfect on their training distribution).',
        'Z2_position_shortcut': 'A model shows a position shortcut if accuracy on shift_1, shift_2 or shift_4 is at least '
                                '20 points below shift_0. Reported as a count; a majority would mean the shortcut '
                                'generalises to the standard benchmark.',
        'Z3_anchored_errors': 'Among errors in shift_2 and shift_4, at least 50% equal the position-anchored token '
                              '(the token where the correct value sits in the unshifted layout) for models showing Z2.',
        'Z4_query_layout': 'Descriptive: near, far, memory_order and reverse_order accuracy relative to shift_0. '
                           'Key-based retrieval predicts all within 10 points.',
    }


def make_cell(examples, seed, prefix, mode):
    if mode == 'upstream':
        tokens, labels = mqar(examples, VOCAB, LENGTH, PAIRS, seed)
        anchored = torch.full_like(labels, -1)
        pair_values = torch.stack([tokens[:, 2 * i + 1] for i in range(PAIRS)], 1)
        return tokens, labels, anchored, pair_values
    rng = np.random.RandomState(seed)
    tokens = np.zeros((examples, LENGTH), dtype=np.int64)
    labels = np.full((examples, LENGTH), -100, dtype=np.int64)
    anchored = np.full((examples, LENGTH), -1, dtype=np.int64)
    pair_values = np.zeros((examples, PAIRS), dtype=np.int64)
    q0 = prefix + 2 * PAIRS
    space = (LENGTH - 1 - q0) // 2 + 1  # query key slots q0 + 2g <= LENGTH - 1
    power = .01 * np.arange(1, space + 1) ** (.01 - 1)
    power /= power.sum()
    for row in range(examples):
        keys = rng.choice(np.arange(1, VOCAB // 2), PAIRS, replace=False)
        values = rng.choice(np.arange(VOCAB // 2, VOCAB), PAIRS, replace=False)
        for i in range(PAIRS):
            tokens[row, prefix + 2 * i], tokens[row, prefix + 2 * i + 1] = keys[i], values[i]
        if mode in ('power', 'memory_order', 'reverse_order'):
            gaps = rng.choice(space, PAIRS, replace=False, p=power)
            if mode == 'memory_order':
                gaps = np.sort(gaps)
            elif mode == 'reverse_order':
                gaps = np.sort(gaps)[::-1]
        elif mode == 'near':
            gaps = rng.permutation(PAIRS)
        elif mode == 'far':
            gaps = space - PAIRS + rng.permutation(PAIRS)
        else:
            raise ValueError(mode)
        for i, gap in enumerate(gaps):
            position = q0 + 2 * gap
            tokens[row, position] = keys[i]
            labels[row, position] = values[i]
            anchored[row, position] = tokens[row, 2 * i + 1]  # where value i sits in the unshifted Zoology layout
        pair_values[row] = values
    return torch.tensor(tokens), torch.tensor(labels), torch.tensor(anchored), torch.tensor(pair_values)


def selected_models(args):
    models = {}
    runs = select_runs(args.root, ['mqar'], args.seeds, [], 'explore')
    for (_, seed, init, variant), (path, _) in runs.items():
        if init in args.initializations:
            models[f'{init}/{variant}/{seed}'] = (path.parent / 'latest_checkpoint.pt', variant, seed, init)
    if args.joint_root:
        from run_joint_recipe_study import identity
        for path in args.joint_root.rglob('summary.json'):
            summary = json.loads(path.read_text())
            cfg = summary.get('settings', {})
            if (cfg.get('protocol') == 'joint-timescale-sensitivity-v1' and cfg.get('family') == 'mqar'
                    and cfg.get('seed') in args.seeds and summary.get('status') == 'complete'
                    and cfg.get('source_hashes') == identity() and (path.parent / 'latest_checkpoint.pt').is_file()):
                models[f"joint/{cfg['variant']}/{cfg['seed']}"] = (path.parent / 'latest_checkpoint.pt', cfg['variant'], cfg['seed'], 'joint')
    for variant in ('scalar', 'channel-wise'):
        path = Path('published_mqar') / variant / 'latest_checkpoint.pt'
        if args.published and path.is_file():
            models[f'published64/{variant}/42'] = (path, variant, 42, 'published64')
    return models


def evaluate(model, tokens, labels, device, batch):
    mask = labels != -100
    predictions = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch):
            predictions.append(model.scored_logits(tokens[start:start + batch].to(device),
                                                   mask[start:start + batch].to(device)).argmax(-1).cpu())
    return torch.cat(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/calibrated_recipe_study/matched-decay-v1'))
    parser.add_argument('--joint-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--initializations', nargs='*', default=['default', 'calibrated'])
    parser.add_argument('--no-published', dest='published', action='store_false')
    parser.add_argument('--examples', type=int, default=512)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--skip-overlap-audit', action='store_true')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    models = selected_models(args)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    data, audits = {}, []
    for seed in sorted({m[2] for m in models.values()}):
        train = None if args.skip_overlap_audit else build_data('mqar', 100000, seed + 100000, 'train', VOCAB, LENGTH)
        for index, (name, prefix, mode) in enumerate(CELLS):
            cell_seed = seed + SEED_OFFSET + 1000 * index
            tokens, labels, anchored, pair_values = make_cell(args.examples, cell_seed, prefix, mode)
            overlap = None
            if train is not None:
                probe = RecipeData(tokens, labels, labels, labels, torch.zeros_like(tokens, dtype=torch.bool), {})
                overlap = split_overlap(train, probe)
                if overlap:
                    raise ValueError(f'Test cell overlaps training data: seed {seed} {name}')
            data[seed, name] = (tokens, labels, anchored, pair_values)
            audits.append(dict(seed=seed, cell=name, cell_seed=cell_seed, prefix=prefix, mode=mode, train_overlap=overlap,
                               identity=manifest(tokens, labels, seed=cell_seed, cell=name)))
        del train
    atomic_write_json(args.output / 'protocol.json', dict(
        purpose=__doc__, hypotheses=hypotheses(), shortcut_gap=SHORTCUT_GAP, seed_offset=SEED_OFFSET, cells=CELLS,
        examples=args.examples, device=str(device), models={k: str(v[0]) for k, v in models.items()}, data=audits))
    print(f'Data ready for {len(models)} models ({time.monotonic() - started:.0f}s)', flush=True)

    results, rows = {}, []
    for name_model, (path, variant, seed, init) in sorted(models.items()):
        model = RecipeLM(variant, VOCAB, LENGTH, 128, 'chunked', seed)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        model = model.to(device).eval()
        for cell, _, _ in CELLS:
            tokens, labels, anchored, pair_values = data[seed, cell]
            predictions = evaluate(model, tokens, labels, device, args.batch)
            mask = labels != -100
            targets, anchor = labels[mask], anchored[mask]
            rows_index = mask.nonzero()[:, 0]
            correct = predictions == targets
            wrong = ~correct
            other_pair = torch.zeros_like(wrong)
            for k in range(PAIRS):
                other_pair |= predictions == pair_values[rows_index, k]
            other_pair &= wrong
            anchored_hit = wrong & (predictions == anchor) & (anchor != targets)
            ordinal = (mask.cumsum(1) - 1)[mask]
            results[name_model, cell] = dict(
                accuracy=float(correct.float().mean()), queries=int(len(correct)), errors=int(wrong.sum()),
                errors_other_memory_value=int(other_pair.sum()), errors_position_anchored=int(anchored_hit.sum()),
                accuracy_by_query_ordinal=[float(correct[ordinal == o].float().mean()) for o in range(PAIRS)])
            rows.append(dict(model=name_model, initialization=init, variant=variant, seed=seed, cell=cell,
                             **{k: v for k, v in results[name_model, cell].items() if k != 'accuracy_by_query_ordinal'}))
        print(f'{name_model}: done ({time.monotonic() - started:.0f}s)', flush=True)

    names = sorted(models)
    acc = lambda m, c: results[m, c]['accuracy']
    shortcut = [m for m in names if min(acc(m, c) for c in ('shift_1', 'shift_2', 'shift_4')) <= acc(m, 'shift_0') - SHORTCUT_GAP]
    anchored = sum(results[m, c]['errors_position_anchored'] for m in shortcut for c in ('shift_2', 'shift_4'))
    errors = sum(results[m, c]['errors'] for m in shortcut for c in ('shift_2', 'shift_4'))
    verdicts = {
        'Z1_sanity': dict(models_at_least_95=sum(acc(m, 'zoology_baseline') >= .95 for m in names), of=len(names)),
        'Z2_position_shortcut': dict(models_showing=len(shortcut), of=len(names), models=shortcut,
                                     majority=len(shortcut) > len(names) / 2),
        'Z3_anchored_errors': dict(anchored_share=anchored / errors if errors else None, errors=errors,
                                   confirmed=bool(errors) and anchored / errors >= .5),
        'Z4_query_layout': {c: dict(models_within_10_points=sum(abs(acc(m, c) - acc(m, 'shift_0')) <= .10 for m in names),
                                    of=len(names)) for c in ('near', 'far', 'memory_order', 'reverse_order')},
    }
    wall = time.monotonic() - started
    atomic_write_json(args.output / 'results.json', dict(status='complete', wall_seconds=wall, verdicts=verdicts,
                      results=[dict(model=m, cell=c, **v) for (m, c), v in results.items()]))
    with (args.output / 'cell_results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Zoology MQAR layout test', '', f'Models: {len(names)}. Wall time {wall:.0f} s. Seed offset {SEED_OFFSET}. '
             'Hypotheses frozen in protocol.json before evaluation.', '', '## Verdicts', '']
    lines += [f'- **{k}**: `{json.dumps(v)}`' for k, v in verdicts.items()]
    lines += ['', '## Accuracy per model and cell', '', '| Model | ' + ' | '.join(c for c, _, _ in CELLS) + ' |',
              '|---|' + '---:|' * len(CELLS)]
    for m in names:
        lines.append(f'| {m} | ' + ' | '.join(f'{acc(m, c):.1%}' for c, _, _ in CELLS) + ' |')
    lines += ['', '## Error attribution (all models)', '', '| Cell | Errors | Another memory value | Position-anchored |',
              '|---|---:|---:|---:|']
    for c, _, _ in CELLS:
        e = sum(results[m, c]['errors'] for m in names)
        lines.append(f"| {c} | {e} | {sum(results[m, c]['errors_other_memory_value'] for m in names)} | "
                     f"{sum(results[m, c]['errors_position_anchored'] for m in names)} |")
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(verdicts, indent=1), flush=True)


if __name__ == '__main__':
    main()
