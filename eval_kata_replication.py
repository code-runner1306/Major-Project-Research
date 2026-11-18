"""Evaluate the kata-replication-v1 runs: replication of KATA Table 2 (GDN, dh=64) and the rewrite-shortcut tests.

Cells (fresh data, seed + 1800000, overlap-checked against the training set):
  replication: two write passes, K in {4, 8, 16, 32} (training range) and {64, 128, 256, 384} (the paper's columns)
  shortcut:    K in {4, 8, 16, 32} with one pass, three passes, four passes, and a third pass for half of the keys
Each wrong answer is attributed to the write of the queried key it came from (write 1, 2, ...), another key's value,
or anything else. Predictions R1-R5 come from run_kata_replication.PREDICTIONS, frozen before training.
"""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from artifacts import atomic_write_json
from kata_model import KataLM
from kata_overwrite_data import OOD_KEYS, TRAIN_KEYS, VOCAB, generate, oracle_check, training_set
from run_kata_replication import PAPER_ROW, PREDICTIONS, PROTOCOL

EVAL_OFFSET = 1800000
SHORTCUT_GAP = .20


def cells(small, large):
    out = [(f'two_pass_K{k}', k, 2, False, small if k <= 32 else large) for k in (*TRAIN_KEYS, *OOD_KEYS)]
    for k in TRAIN_KEYS:
        out += [(f'one_pass_K{k}', k, 1, False, small), (f'three_pass_K{k}', k, 3, False, small),
                (f'four_pass_K{k}', k, 4, False, small), (f'partial_third_K{k}', k, 3, True, small)]
    return out


def attribute(tokens, labels, predictions):
    """Per query: (outcome, number of writes of that key)."""
    results = []
    for row_tokens, row_labels, row_predictions in zip(tokens.tolist(), labels.tolist(), predictions):
        history, value_owner = {}, {}
        answers = iter(row_predictions)
        for position, label in enumerate(row_labels):
            token = row_tokens[position]
            if label != -100:
                answer, writes = next(answers), history[token]
                if answer == writes[-1]:
                    outcome = 'correct'
                elif answer in writes:
                    outcome = f'write_{writes.index(answer) + 1}'
                elif answer in value_owner:
                    outcome = 'other_key_value'
                else:
                    outcome = 'other'
                results.append((outcome, len(writes)))
            elif 0 < token < VOCAB // 2 and position + 1 < len(row_tokens) and row_tokens[position + 1] >= VOCAB // 2:
                history.setdefault(token, []).append(row_tokens[position + 1])
                value_owner[row_tokens[position + 1]] = token
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/kata_replication'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--small-examples', type=int, default=512)
    parser.add_argument('--large-examples', type=int, default=256)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--allow-smoke', action='store_true', help='Accept CPU smoke runs (testing only)')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    runs = {}
    for path in args.root.rglob('summary.json'):
        summary = json.loads(path.read_text())
        settings = summary.get('settings', {})
        if settings.get('protocol') != PROTOCOL or summary.get('status') != 'complete':
            continue
        if settings['epochs'] != 32 and not args.allow_smoke:
            continue
        key = f"lr_{settings['learning_rate']:g}"
        if key in runs:
            raise ValueError(f'Duplicate run for {key}; point --root at one session')
        runs[key] = (path.parent / 'latest_checkpoint.pt', summary)
    if not runs:
        raise ValueError(f'No complete {PROTOCOL} runs under {args.root}')
    seed = {s['settings']['seed'] for _, s in runs.values()}
    if len(seed) != 1:
        raise ValueError('Runs use different seeds')
    seed = seed.pop()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    train_rows = {}
    for k, (t, l) in training_set(100000, seed).items():
        train_rows[k] = {hashlib.sha256(torch.cat([a, b]).numpy().tobytes()).digest() for a, b in zip(t, l)}
    data, audits = {}, []
    for name, k, passes, partial, n in cells(args.small_examples, args.large_examples):
        tokens, labels, _ = generate(n, k, passes, seed + EVAL_OFFSET, 'evaluation', partial)
        overlap = 0
        if passes == 2 and k in train_rows:
            overlap = sum(hashlib.sha256(torch.cat([a, b]).numpy().tobytes()).digest() in train_rows[k]
                          for a, b in zip(tokens, labels))
        failures = oracle_check(tokens, labels)
        if overlap or failures:
            raise ValueError(f'Evaluation data audit failed for {name}: overlap {overlap}, oracle failures {failures}')
        data[name] = (tokens, labels)
        audits.append(dict(cell=name, keys=k, passes=passes, partial_third=partial, examples=n, train_overlap=overlap,
                           oracle_failures=failures, length=tokens.shape[1]))
    del train_rows
    atomic_write_json(args.output / 'protocol.json', dict(purpose=__doc__, predictions=PREDICTIONS, paper_row=PAPER_ROW,
        eval_offset=EVAL_OFFSET, seed=seed, device=str(device), runs={k: str(v[0]) for k, v in runs.items()}, data=audits))
    print(f'Data ready ({time.monotonic() - started:.0f}s)', flush=True)

    results = {}
    for key, (path, summary) in sorted(runs.items()):
        model = KataLM(seed=seed)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'], strict=True)
        model = model.to(device).eval()
        for name, (tokens, labels) in data.items():
            batch = max(1, 16384 // tokens.shape[1])
            predictions = []
            with torch.no_grad():
                for start in range(0, len(tokens), batch):
                    t, l = tokens[start:start + batch].to(device), labels[start:start + batch].to(device)
                    mask = l != -100
                    flat = model.scored_logits(t, mask).argmax(-1).cpu()
                    counts = mask.sum(1).tolist()
                    predictions += [p.tolist() for p in torch.split(flat, counts)]
            outcomes = attribute(tokens, labels, predictions)
            by_writes = {}
            for outcome, writes in outcomes:
                by_writes.setdefault(writes, Counter())[outcome] += 1
            total = Counter(o for o, _ in outcomes)
            results[key, name] = dict(accuracy=total['correct'] / len(outcomes), outcomes=dict(total),
                                      accuracy_by_key_writes={w: c['correct'] / sum(c.values()) for w, c in sorted(by_writes.items())},
                                      outcomes_by_key_writes={w: dict(c) for w, c in sorted(by_writes.items())})
        print(f'{key}: done ({time.monotonic() - started:.0f}s)', flush=True)

    models = sorted(runs)
    acc = lambda m, c: results[m, c]['accuracy']
    paper_columns = [f'two_pass_K{k}' for k in OOD_KEYS]
    selected = max(models, key=lambda m: mean(acc(m, c) for c in paper_columns))
    group = lambda m, prefix: mean(acc(m, f'{prefix}_K{k}') for k in TRAIN_KEYS)
    shortcut = {m: group(m, 'three_pass') <= group(m, 'two_pass') - SHORTCUT_GAP for m in models}
    errors = Counter()
    for m in models:
        for k in TRAIN_KEYS:
            errors.update({o: n for o, n in results[m, f'three_pass_K{k}']['outcomes'].items() if o != 'correct'})
    second = errors['write_2'] / max(1, sum(errors.values()))
    partial = {}
    for m in models:
        rewritten = [results[m, f'partial_third_K{k}']['accuracy_by_key_writes'].get(3) for k in TRAIN_KEYS]
        kept = [results[m, f'partial_third_K{k}']['accuracy_by_key_writes'].get(2) for k in TRAIN_KEYS]
        partial[m] = dict(rewritten_keys=mean(rewritten), keys_left_at_pass_2=mean(kept))
    verdicts = {
        'R1_replication': dict(selected_learning_rate=selected,
                               ours={c.replace('two_pass_', ''): round(acc(selected, c), 3) for c in paper_columns},
                               ours_average=round(mean(acc(selected, c) for c in paper_columns), 3), paper=PAPER_ROW,
                               adequate=acc(selected, 'two_pass_K64') >= .95),
        'R2_rewrite_shortcut': dict(selected_shows_it=shortcut[selected], all_models={m: dict(
            two_pass=round(group(m, 'two_pass'), 3), three_pass=round(group(m, 'three_pass'), 3),
            four_pass=round(group(m, 'four_pass'), 3), shortcut=shortcut[m]) for m in models}),
        'R3_error_identity': dict(second_write_share=round(second, 3), error_counts=dict(errors), confirmed=second >= .5),
        'R4_partial_third_pass': {m: {k: round(v, 3) for k, v in d.items()} for m, d in partial.items()},
        'R5_single_pass': {m: round(group(m, 'one_pass'), 3) for m in models},
    }
    wall = time.monotonic() - started
    atomic_write_json(args.output / 'results.json', dict(status='complete', wall_seconds=wall, verdicts=verdicts,
                      results=[dict(model=m, cell=c, **v) for (m, c), v in results.items()],
                      training={m: dict(epochs_run=s['epochs_run'], stopped_early=s['stopped_early'], final=s['final'])
                                for m, (_, s) in runs.items()}))
    with (args.output / 'cell_results.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'cell', 'accuracy', 'outcomes'])
        for (m, c), v in sorted(results.items()):
            writer.writerow([m, c, v['accuracy'], json.dumps(v['outcomes'])])
    names = [c[0] for c in cells(0, 0)]
    lines = ['# KATA replication and rewrite-shortcut evaluation', '',
             f'Runs: {", ".join(models)}. Wall time {wall:.0f} s. Predictions frozen before training.', '',
             '## Verdicts', ''] + [f'- **{k}**: `{json.dumps(v)}`' for k, v in verdicts.items()]
    lines += ['', '## Accuracy per cell', '', '| Cell | ' + ' | '.join(models) + ' |', '|---|' + '---:|' * len(models)]
    lines += [f'| {c} | ' + ' | '.join(f'{acc(m, c):.1%}' for m in models) + ' |' for c in names]
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(verdicts, indent=1), flush=True)


if __name__ == '__main__':
    main()
