"""The hidden block-first shortcut: saved-results summary, local overwrite models, live evaluation.

Read-only with respect to research code and result files. Every number in summary() is parsed from the saved
artifacts under research_results/ (nothing is hard-coded). evaluate() runs the local overwrite checkpoints on fresh
examples; the data seed offset is the one app.py sets on confirm_block_layout (SEED_OFFSET). Patched runs replace the
GDN forward methods through analyze_write_gate.instrument and always restore them in a finally block, so cached
models are left as loaded. The caller (app.py) holds the model lock around evaluate().
"""
import csv
import json
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_write_gate as awg  # noqa: E402
import confirm_block_layout as cbl  # noqa: E402
from checkpoints import repo_relative  # noqa: E402

RESULTS = ROOT / 'research_results'
BLOCK_DIR = RESULTS / 'block_layout_confirmation_20261117'
MECHANISM_DIR = RESULTS / 'write_gate_mechanism_20261117'
STRESS_DIR = RESULTS / 'memory_stress_tests_20261117'
MODEL_ROOT = ROOT / 'output' / 'calibrated_recipe_study' / 'matched-decay-v1'

SEEDS = (42, 43, 44)
VARIANTS = ('scalar', 'channel-wise')
FOLDER_TO_INIT = {'default': 'default', 'calibrated': 'mean-decay'}   # folder label -> API label
INIT_TO_FOLDER = {label: folder for folder, label in FOLDER_TO_INIT.items()}
PATCH_CELL = 'three_rewrites'          # write_gate_mechanism runs patches on this cell
PATCHES = ('none', 'beta_one_latest', 'key_align_latest', 'beta_zero_rewrite1')
OUTCOME_TO_SOURCE = {'correct': 'latest', 'initial': 'initial', 'first_rewrite': 'first_rewrite',
                     'second_rewrite': 'second_rewrite', 'third_rewrite': 'third_rewrite', 'other': 'other'}
SOURCES = tuple(OUTCOME_TO_SOURCE.values())
BATCH = 64
N_LAYERS = 4
EXAMPLES_SHOWN = 3          # default for show_examples
EXAMPLES_MAX = 8


class ShortcutError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _init_label(folder_or_label):
    return FOLDER_TO_INIT.get(folder_or_label, folder_or_label)


# ------------------------------------------------------------------------------ local models

@lru_cache(maxsize=1)
def _local_checkpoints():
    """{(init_label, variant, seed): checkpoint Path} for the overwrite runs present on disk."""
    found = {}
    for path in sorted(MODEL_ROOT.glob('*/overwrite/seed_*/*/*/latest_checkpoint.pt')):
        session, family, seed_dir, folder, variant = path.relative_to(MODEL_ROOT).parts[:5]
        seed = int(seed_dir.split('_', 1)[1])
        if folder not in FOLDER_TO_INIT or variant not in VARIANTS or seed not in SEEDS:
            continue
        key = (_init_label(folder), variant, seed)
        if key in found:
            raise RuntimeError(f'Ambiguous overwrite checkpoints for {key}: {found[key]} and {path}')
        found[key] = path
    return found


def local_checkpoints():
    return dict(_local_checkpoints())


def models():
    """The local overwrite models, in (init, variant, seed) order."""
    return [dict(init=init, variant=variant, seed=seed, checkpoint=repo_relative(path))
            for (init, variant, seed), path in sorted(_local_checkpoints().items())]


@lru_cache(maxsize=12)
def _load(path_text, variant, seed):
    """Eval-mode RecipeLM on CPU, cached (at most 12 resident). Shared: callers must not leave it patched."""
    return awg.load(Path(path_text), variant, seed)


# ----------------------------------------------------------------------------- summary

def _threshold_text(label, statement, thresholds):
    gap = thresholds['shortcut_gap'] * 100
    special = thresholds['special_match'] * 100
    if label in ('H2', 'H3'):
        match = re.search(r'at least (\d+)%', statement)
        pct = match.group(1) if match else '?'
    if label == 'H1':
        return f'block_first latest accuracy at least {gap:.0f} points below general_nonblock (main_r3); majority of models'
    if label == 'H2':
        return f'first-replacement share of block_first errors at least {pct}%'
    if label == 'H3':
        return f'block_first latest accuracy at least {pct}% in all but at most one model (main_r1)'
    if label == 'H4':
        return f'within {special:.0f} points of general_nonblock in a majority of shortcut models'
    return f'block_first deficit of at least {gap:.0f} points persists at query distance 12 and 16 (open question)'


def _per_model(thresholds):
    rows = list(csv.DictReader((BLOCK_DIR / 'cell_results.csv').read_text(encoding='utf-8').splitlines()))
    table = {}
    for row in rows:
        key = (_init_label(row['initialization']), row['variant'], int(row['seed']))
        table.setdefault(key, {})[(row['cell'], row['layout'])] = float(row['latest_value_accuracy'])
    local = set(_local_checkpoints())
    out = []
    for key in sorted(table):
        init, variant, seed = key
        cells = table[key]
        block_r3 = cells[('main_r3', 'block_first')]
        general_r3 = cells[('main_r3', 'general_nonblock')]
        out.append(dict(
            init=init, variant=variant, seed=seed, local=key in local,
            block_first_r3=block_r3, general_r3=general_r3,
            block_first_r1=cells[('main_r1', 'block_first')],
            shortcut=(general_r3 - block_r3) >= thresholds['shortcut_gap']))
    return out


def _rule_agreement(block):
    rows = block.get('rule_agreement')
    if rows is None:
        return None, 'results.json has no rule_agreement; predictions.csv.gz not parsed (not required)'
    out = []
    for row in rows:
        init, variant, seed = row['model'].split('/')
        out.append(dict(init=_init_label(init), variant=variant, seed=int(seed),
                        **{k: v for k, v in row.items() if k != 'model'}))
    return out, None


def _patches():
    results = _json(MECHANISM_DIR / 'results.json')['results']
    tables_path = MECHANISM_DIR / 'tables.md'
    material = None
    if tables_path.exists():
        match = re.search(r'>=\s*(\d+)\s*points', tables_path.read_text(encoding='utf-8'))
        material = int(match.group(1)) / 100 if match else None
    cell_rows = [r for r in results if r['cell'] == PATCH_CELL and r['patch_layers'] == 'all']
    base = {(r['initialization'], r['variant'], r['seed']): r for r in cell_rows if r['patch'] == 'none'}
    order, by_patch = [], {}
    for row in cell_rows:
        if row['patch'] == 'none':
            continue
        if row['patch'] not in by_patch:
            order.append(row['patch'])
        by_patch.setdefault(row['patch'], []).append(row)
    out = []
    for patch in order:
        deltas = []
        for row in by_patch[patch]:
            b = base[(row['initialization'], row['variant'], row['seed'])]
            deltas.append(dict(latest=row['latest_value_accuracy'] - b['latest_value_accuracy'],
                               unrelated=row['unrelated_retention'] - b['unrelated_retention'],
                               first=row['first_rewrite_error_rate'] - b['first_rewrite_error_rate']))
        n = len(deltas)
        out.append(dict(
            patch=patch, cell=PATCH_CELL, models=n,
            latest_delta=sum(d['latest'] for d in deltas) / n,
            unrelated_delta=sum(d['unrelated'] for d in deltas) / n,
            first_rewrite_delta=sum(d['first'] for d in deltas) / n,
            models_up=None if material is None else sum(d['latest'] >= material for d in deltas),
            models_down=None if material is None else sum(d['latest'] <= -material for d in deltas),
            material_change=material))
    return out


def _stress_first_sign():
    rows = _json(STRESS_DIR / 'error_breakdown.json')['overwrite']
    errors = sum(r['total_latest'] - r['correct_latest'] for r in rows)
    first = sum(r['first_rewrite_errors'] for r in rows)
    return dict(models=len(rows), newest_value_errors=errors,
                first_rewrite_errors=first,
                initial_value_errors=sum(r['initial_value_errors'] for r in rows),
                second_rewrite_errors=sum(r['second_rewrite_errors'] for r in rows),
                other_errors=sum(r['other_errors'] for r in rows),
                first_rewrite_share=first / errors if errors else None,
                source=repo_relative(STRESS_DIR / 'error_breakdown.json'))


def _pct1(x):
    return f'{x * 100:.1f}%'


def _distance_of(key):
    """'distance_12' -> 12, for the H5 keys."""
    return int(key.split('_', 1)[1])


def _readout(label, observed, statement):
    """Plain-English sentence, a 0-1 observed value and a 0-1 threshold value for one hypothesis.

    Every count comes from the parsed `observed` dict (results.json). Only the threshold fractions that the
    protocol states in prose (H2's 80%, H3's "all but at most one") are read from the statement or the rule itself.
    """
    if label == 'H1':
        show, of = observed['models_showing'], observed['of']
        return (f'{show} of {of} models show the shortcut.', show / of, 0.5)
    if label == 'H2':
        counts = observed['error_counts']
        total = sum(counts.values())
        first = counts['first_rewrite']
        match = re.search(r'at least (\d+)%', statement)
        pct = int(match.group(1)) / 100 if match else 0.8
        return (f'{_pct1(first / total)} of block-first errors are the first replacement '
                f'({first:,} of {total:,}).', first / total, pct)
    if label == 'H3':
        at_least, of = observed['models_at_least_95'], observed['of']
        return (f'{at_least} of {of} models score at least 95% on the training layout.',
                at_least / of, (of - 1) / of)
    if label == 'H4':
        by_layout = observed['by_layout']
        shares = {name: v['models_within_10_points'] / v['of'] for name, v in by_layout.items()}
        pairs = {(v['models_within_10_points'], v['of']) for v in by_layout.values()}
        if len(pairs) == 1:
            within, of = pairs.pop()
            text = f'Each near-miss layout behaves normally in {within} of {of} shortcut models.'
        else:
            lows = [v['models_within_10_points'] for v in by_layout.values()]
            highs = [v['of'] for v in by_layout.values()]
            text = (f'Near-miss layouts behave normally in {min(lows)} to {max(lows)} of '
                    f'{min(highs)} to {max(highs)} shortcut models.')
        return (text, min(shares.values()), 0.5)
    if label == 'H5':
        parts, shares = [], []
        for key in sorted(observed, key=_distance_of):
            v = observed[key]
            parts.append(f'at distance {_distance_of(key)} in {v["deficit_persists"]} of {v["of"]}')
            shares.append(v['deficit_persists'] / v['of'])
        return (f'The gap persists {" and ".join(parts)} shortcut models.', min(shares), 0.5)
    raise ValueError(f'no plain-English reading defined for hypothesis {label}')


@lru_cache(maxsize=1)
def summary():
    block_protocol = _json(BLOCK_DIR / 'protocol.json')
    block = _json(BLOCK_DIR / 'results.json')
    # The saved file holds the flat all-models verdicts; newer scripts wrap them as {"all_models": {...}, "group_*": ...}.
    verdicts = block['verdicts'].get('all_models', block['verdicts'])
    thresholds = block_protocol['thresholds']
    hypotheses = {}
    for key, statement in block_protocol['hypotheses'].items():
        label = key.split('_', 1)[0]
        verdict = verdicts[key]
        observed = {k: v for k, v in verdict.items() if k != 'confirmed'}
        if 'confirmed' in verdict:
            confirmed = bool(verdict['confirmed'])
        else:
            # H5 is an open question with no saved boolean: derived here as "deficit persists in a majority of
            # shortcut models at every distance" (persists counts are in observed).
            confirmed = all(v['deficit_persists'] > v['of'] / 2 for v in observed.values())
        observed_text, observed_value, threshold_value = _readout(label, observed, statement)
        hypotheses[label] = dict(
            statement=statement,
            threshold=_threshold_text(label, statement, thresholds),
            threshold_value=threshold_value,
            observed=observed,
            observed_text=observed_text,
            observed_value=observed_value,
            confirmed=confirmed,
            confirmed_source='results.json' if 'confirmed' in verdict else 'derived (no saved boolean)')
    rule_rows, rule_reason = _rule_agreement(block)
    per_model = _per_model(thresholds)
    provenance_files = {
        'hypotheses': repo_relative(BLOCK_DIR / 'protocol.json'),
        'verdicts': repo_relative(BLOCK_DIR / 'results.json'),
        'per_model': repo_relative(BLOCK_DIR / 'cell_results.csv'),
        'rule_agreement': repo_relative(BLOCK_DIR / 'results.json'),
        'patches': repo_relative(MECHANISM_DIR / 'results.json'),
        'patch_material_threshold': repo_relative(MECHANISM_DIR / 'tables.md'),
        'stress_first_sign': repo_relative(STRESS_DIR / 'error_breakdown.json'),
        'report_not_parsed': repo_relative(BLOCK_DIR / 'REPORT.md'),
    }
    return dict(
        hypotheses=hypotheses,
        per_model=per_model,
        rule_agreement=rule_rows,
        **({'rule_agreement_reason': rule_reason} if rule_reason else {}),
        patches=_patches(),
        stress_first_sign=_stress_first_sign(),
        provenance=dict(files=provenance_files, block_layout_wall_seconds=block.get('wall_seconds'),
                        shortcut_gap=thresholds['shortcut_gap'], special_match=thresholds['special_match']),
    )


# --------------------------------------------------------------------------- evaluate

def allowed_layouts(rewrites):
    return {'general_nonblock', *cbl.special_patterns(rewrites)}


@lru_cache(maxsize=32)
def _sample(example_seed, distance, rewrites, layout, examples):
    chosen, _ = cbl.sample_layouts(example_seed, 0, 4, distance, rewrites, [layout], examples)
    return tuple(chosen[layout])


def roles_for(record, tokens):
    """Identical to app.roles_for (layout-fix try endpoint); test_shortcut checks the two agree."""
    md = record.metadata
    roles = ['filler'] * len(tokens)
    for p in md['all_write_positions']:
        if p < len(tokens):
            roles[p] = 'value'
            roles[p - 1] = 'key'
    for p in md['write_positions']:
        if p < len(tokens):
            roles[p] = 'target_value'
            roles[p - 1] = 'target_key'
    for a in record.answers:
        roles[a.label_position] = 'query_latest' if a.kind == 'latest' else 'query_unrelated'
        if a.label_position - 1 >= 0 and tokens[a.label_position - 1] == 0:
            roles[a.label_position - 1] = 'marker'
    return roles


def _latest_answers(predictions, cell):
    mask = cell['case'].data.labels != -100
    answers = []
    for i, row in enumerate(torch.split(predictions, mask.sum(1).tolist())):
        lookup = dict(zip(mask[i].nonzero().flatten().tolist(), row.tolist()))
        answers.append(lookup[cell['positions'].latest_label[i]])
    return answers


def _slot_histogram(records, outcomes, answers):
    """Wrong latest answers: which memory slot (index in all_write_positions) returned the value.

    If the returned value occurs in several slots, the most recent matching slot is used. Tokens between are
    counted from the slot's value token to the query marker (exclusive of both).
    """
    counts, unmatched = Counter(), 0
    for record, outcome, answer in zip(records, outcomes, answers):
        if outcome == 'correct':
            continue
        md = record.metadata
        slots = md['all_write_positions']
        targets = set(md['write_positions'])
        match = None
        for slot in range(len(slots) - 1, -1, -1):
            p = slots[slot]
            if p < len(record.tokens) and int(record.tokens[p]) == answer:
                match = slot
                break
        if match is None:
            unmatched += 1
            continue
        p = slots[match]
        label = next(a.label_position for a in record.answers if a.kind == 'latest')
        marker = label - 1 if label - 1 >= 0 and int(record.tokens[label - 1]) == 0 else label
        counts[(match, p in targets, marker - p - 1)] += 1
    histogram = [dict(slot=slot, is_target=is_target, count=count, tokens_before_query=between)
                 for (slot, is_target, between), count in sorted(counts.items())]
    return histogram, unmatched


def _writes(record):
    """Memory as a list in write order (1-based slot). Key = token before the value, as in the layout fix."""
    md = record.metadata
    targets = set(md['write_positions'])
    latest = md['write_positions'][-1]
    tokens = record.tokens
    return [dict(slot=slot, key=int(tokens[p - 1]), value=int(tokens[p]),
                 is_target=p in targets, is_latest_target=p == latest)
            for slot, p in enumerate(md['all_write_positions'], start=1)]


def _picked_slot(writes, answer):
    """1-based slot whose value equals the answer; the most recent such slot. None if no slot holds it."""
    for row in reversed(writes):
        if row['value'] == answer:
            return row['slot']
    return None


def _pick_histogram(writes_per, picked, n):
    """Counts of the slot each latest-value answer came from, over all n answers.

    Slot flags are majority votes across examples (fixed-rewrite layouts make them identical in every example).
    Answers that match no memory slot are counted under slot None.
    """
    width = max(len(w) for w in writes_per)
    counts = [0] * width
    target_votes, latest_votes = [0] * width, [0] * width
    for writes in writes_per:
        for row in writes:
            i = row['slot'] - 1
            target_votes[i] += row['is_target']
            latest_votes[i] += row['is_latest_target']
    unmatched = 0
    for slot in picked:
        if slot is None:
            unmatched += 1
        else:
            counts[slot - 1] += 1
    histogram = [dict(slot=i + 1, count=counts[i],
                      is_target=target_votes[i] * 2 > n, is_latest_target=latest_votes[i] * 2 > n)
                 for i in range(width)]
    histogram.append(dict(slot=None, count=unmatched))
    return histogram


def _attribution(recorded, cell, outcomes):
    """Per layer: mean share of the readout (analyze_write_gate.attribution) that sits on the target writes'
    key and value tokens, split by latest-query outcome. Same quantity as diagnostics' role_target_writes rows,
    computed directly because diagnostics() also builds ~20 unused rows per example and is too slow."""
    p = cell['positions']
    shares = {layer: {'correct': [], 'wrong': []} for layer in range(N_LAYERS)}
    offset = 0
    for batch_records in recorded:
        size = batch_records[0]['beta'].size(0)
        for layer, r in enumerate(batch_records):
            for i in range(size):
                ex = offset + i
                targets = sorted(set(p.key[ex]) | set(p.value[ex]))
                weights = awg.attribution(r['q'][i] * r['q'].size(-1) ** -.5, r['k'][i], r['alpha'][i],
                                          r['beta'][i], p.latest_label[ex])
                total = weights.abs().sum().clamp_min(1e-12)
                bucket = 'correct' if outcomes[ex] == 'correct' else 'wrong'
                shares[layer][bucket].append(float(weights[targets].abs().sum() / total))
        offset += size
    out = []
    for layer in range(N_LAYERS):
        correct, wrong = shares[layer]['correct'], shares[layer]['wrong']
        out.append(dict(layer=layer + 1,
                        target_share_correct=sum(correct) / len(correct) if correct else None,
                        target_share_wrong=sum(wrong) / len(wrong) if wrong else None))
    return out


@torch.no_grad()
def evaluate(init, variant, seed, layout, rewrites, distance, examples, example_seed, patch,
             show_examples=EXAMPLES_SHOWN):
    """Run one local overwrite model on fresh examples. Caller must hold the model lock."""
    if layout not in allowed_layouts(rewrites):
        raise ShortcutError(422, f"layout '{layout}' is not valid for rewrites={rewrites}")
    if patch not in PATCHES:
        raise ShortcutError(422, f"unknown patch '{patch}'")
    if not 1 <= show_examples <= EXAMPLES_MAX:
        raise ShortcutError(422, f'show_examples must be 1..{EXAMPLES_MAX}')
    path = _local_checkpoints().get((init, variant, seed))
    if path is None:
        raise ShortcutError(404, f'no local overwrite checkpoint for {init}/{variant}/seed {seed}')
    started = time.perf_counter()
    try:
        records = list(_sample(example_seed, distance, rewrites, layout, examples))
    except (RuntimeError, ValueError) as exc:
        raise ShortcutError(422, f'could not sample this layout: {exc}')
    cell = cbl.to_cell(records, rewrites, dict(example_seed=example_seed, distance=distance, layout=layout))
    model = _load(str(path), variant, seed)
    device = torch.device('cpu')
    rules_per = [cbl.rule_answers(record) for record in records]

    attribution = None
    if patch == 'none':
        outcomes, unrelated, answers = cbl.evaluate(model, cell, device, BATCH)
    else:
        try:
            ctx = awg.instrument(model)
            ctx.patch, ctx.layers = patch, set(range(N_LAYERS))
            predictions, recorded = awg.predict(model, cell, ctx, BATCH, record=True)
            outcomes, unrelated = awg.classify(predictions, cell)
            answers = _latest_answers(predictions, cell)
            attribution = _attribution(recorded, cell, outcomes)
        finally:
            awg.restore(model)

    n = len(outcomes)
    counts = Counter(outcomes)
    sources = {name: 0 for name in SOURCES}
    for outcome in outcomes:
        sources[OUTCOME_TO_SOURCE[outcome]] += 1
    errors = n - counts['correct']
    histogram, unmatched = _slot_histogram(records, outcomes, answers)
    writes_per = [_writes(record) for record in records]
    picked = [_picked_slot(writes_per[i], answers[i]) for i in range(n)]
    pick_hist = _pick_histogram(writes_per, picked, n)
    shown = []
    for i in range(min(show_examples, n)):
        tokens = [int(t) for t in cell['tokens'][i].tolist()]
        prediction = answers[i]
        label = next(a.label_position for a in records[i].answers if a.kind == 'latest')
        latest_row = next(row for row in writes_per[i] if row['is_latest_target'])
        shown.append(dict(
            tokens=tokens, roles=roles_for(records[i], tokens), pattern=cbl.pattern_of(records[i]),
            write_values=[int(v) for v in records[i].metadata['write_values']],
            gold=rules_per[i]['latest'], prediction=prediction,
            which_write=OUTCOME_TO_SOURCE[outcomes[i]],
            rules_matched=[rule for rule in cbl.RULES if rules_per[i][rule] == prediction],
            writes=writes_per[i], question_key=int(records[i].metadata['target_key']),
            picked_slot=picked[i], gold_slot=latest_row['slot'],
            tokens_between_last_write_and_question=label - 2 - records[i].metadata['write_positions'][-1]))
    return dict(
        model=dict(init=init, variant=variant, seed=seed, checkpoint=repo_relative(path)),
        layout=layout, rewrites=rewrites, distance=distance, example_seed=example_seed, patch=patch,
        n_examples=n,
        accuracy_latest=counts['correct'] / n,
        accuracy_unrelated=sum(unrelated) / n,
        first_rewrite_rate=counts['first_rewrite'] / errors if errors else None,
        answer_sources={name: count / n for name, count in sources.items()},
        rule_agreement={rule: sum(rules_per[i][rule] == answers[i] for i in range(n)) / n for rule in cbl.RULES},
        slot_histogram=histogram,
        pick_histogram=pick_hist,
        unmatched_wrong=unmatched,
        attribution=attribution,
        examples=shown,
        elapsed_seconds=time.perf_counter() - started,
    )
