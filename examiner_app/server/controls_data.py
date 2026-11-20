"""Negative controls page: saved Zoology MQAR and KATA results, plus a live Zoology try-it (read-only).

Saved results are parsed from research_results/zoology_layout_test_20261118 and
research_results/kata_replication_eval_20261118_103946; nothing is hard-coded. Zoology checkpoints are listed from
the test's own protocol.json and checked on disk. KATA checkpoints are not on this machine (output/kata_replication
holds logs and summaries only), so KATA is reported as results only.

The Zoology try-it uses zoology_layout_test.make_cell (the saved test's own generator) and the same forward pass and
scoring as the test. Slot picking follows shortcut._picked_slot. Seeds: cell seed = example_seed + 1,600,000 +
1000 * cell index, so example_seed 42 with 512 examples reproduces the saved cell for seed 42 exactly.
"""
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import analyze_write_gate as awg  # noqa: E402
import zoology_layout_test as zlt  # noqa: E402
import shortcut  # noqa: E402
from checkpoints import repo_relative  # noqa: E402

RESULTS = ROOT / 'research_results'
ZOOLOGY_DIR = RESULTS / 'zoology_layout_test_20261118'
KATA_DIR = RESULTS / 'kata_replication_eval_20261118_103946'
KATA_OUTPUT = ROOT / 'output' / 'kata_replication'

INIT_LABEL = {'default': 'default', 'calibrated': 'mean-decay', 'published64': 'published'}
DESIGN_INITS = ('default', 'mean-decay')
VARIANTS = ('scalar', 'channel-wise')
SEEDS = (42, 43, 44)
TRY_MAX_EXAMPLES = 64          # the endpoint limit; the saved test used 512
BATCH = 128
PAIRS = zlt.PAIRS
CELL_NAMES = [name for name, _, _ in zlt.CELLS]
CELL_INDEX = {name: i for i, name in enumerate(CELL_NAMES)}
SHORTCUT_GAP = zlt.SHORTCUT_GAP

LAYOUT_TEXT = {
    'zoology_baseline': ('Standard layout (upstream generator)', 'The layout the models were trained on, from the upstream generator.'),
    'shift_0': ('Standard layout, fresh examples', 'Our copy of the standard generator, with new examples.'),
    'shift_1': ('1 blank token before the table', 'One blank token shifts the table, so keys and values swap positions.'),
    'shift_2': ('2 blank tokens before the table', 'Two blank tokens before the table.'),
    'shift_4': ('4 blank tokens before the table', 'Four blank tokens before the table.'),
    'shift_8': ('8 blank tokens before the table', 'Eight blank tokens before the table.'),
    'shift_16': ('16 blank tokens before the table', 'Sixteen blank tokens before the table.'),
    'near': ('All questions near the table', 'Every question is asked in the first available slots.'),
    'far': ('All questions far from the table', 'Every question is asked in the last available slots.'),
    'memory_order': ('Questions in table order', 'Questions asked in the same order as the table.'),
    'reverse_order': ('Questions in reverse table order', 'Questions asked in the reverse order of the table.'),
    'shift_8_far': ('8 blank tokens and far questions', 'Eight blank tokens before the table and far questions together.'),
}
TRY_SEED_OFFSET = zlt.SEED_OFFSET


class ControlsError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _csv(path):
    return list(csv.DictReader(path.read_text(encoding='utf-8').splitlines()))


def _pct(x, digits=1):
    return f'{x * 100:.{digits}f}%'


# ------------------------------------------------------------------------ Zoology models

def _zoology_models():
    """All 14 models in the saved test, from its protocol.json, with a live check that each file exists."""
    protocol = _json(ZOOLOGY_DIR / 'protocol.json')
    out = []
    for key, raw in protocol['models'].items():
        init_raw, variant, seed = key.split('/')
        path = ROOT / raw.replace('\\', '/')
        out.append(dict(init=INIT_LABEL[init_raw], variant=variant, seed=int(seed),
                        group='published' if init_raw == 'published64' else 'main',
                        local=path.is_file(), path=path, checkpoint=repo_relative(path)))
    order = {'default': 0, 'mean-decay': 1, 'published': 2}
    out.sort(key=lambda m: (order[m['init']], VARIANTS.index(m['variant']), m['seed']))
    return out


def zoology_models():
    return [{k: v for k, v in m.items() if k != 'path'} for m in _zoology_models()]


def _find_zoology(init, variant, seed):
    for m in _zoology_models():
        if (m['init'], m['variant'], m['seed']) == (init, variant, seed):
            return m
    raise ControlsError(404, f'no Zoology model {init}/{variant}/seed {seed} in the saved test')


@lru_cache(maxsize=4)
def _load(path_text, variant, seed):
    """Eval-mode RecipeLM on CPU, cached (at most four resident). Never patched, so safe to share."""
    return awg.load(Path(path_text), variant, seed)


# ------------------------------------------------------------------------ saved summary

@lru_cache(maxsize=1)
def _zoology_saved_rows():
    rows = _csv(ZOOLOGY_DIR / 'cell_results.csv')
    table = {}
    for row in rows:
        table.setdefault(row['model'], {'init': INIT_LABEL[row['initialization']], 'variant': row['variant'],
                                        'seed': int(row['seed']), 'cells': {}})['cells'][row['cell']] = dict(
            accuracy=float(row['accuracy']), errors=int(row['errors']),
            errors_other_memory_value=int(row['errors_other_memory_value']),
            errors_position_anchored=int(row['errors_position_anchored']), queries=int(row['queries']))
    return table


def _zoology_summary():
    results = _json(ZOOLOGY_DIR / 'results.json')
    protocol = _json(ZOOLOGY_DIR / 'protocol.json')
    verdicts = results['verdicts']
    models_by_key = {(m['init'], m['variant'], m['seed']): m for m in _zoology_models()}
    saved = _zoology_saved_rows()
    model_rows = []
    for key, row in saved.items():
        meta = models_by_key[(row['init'], row['variant'], row['seed'])]
        accuracy = {c: v['accuracy'] for c, v in row['cells'].items()}
        # Same rule as the saved test (Z2): a shifted layout at least SHORTCUT_GAP below the standard one.
        shortcut_flag = min(accuracy[c] for c in ('shift_1', 'shift_2', 'shift_4')) <= accuracy['shift_0'] - SHORTCUT_GAP
        model_rows.append(dict(init=row['init'], variant=row['variant'], seed=row['seed'], group=meta['group'],
                               local=meta['local'], checkpoint=meta['checkpoint'], shortcut=shortcut_flag,
                               accuracy=accuracy))
    model_rows.sort(key=lambda r: ({'default': 0, 'mean-decay': 1, 'published': 2}[r['init']],
                                   VARIANTS.index(r['variant']), r['seed']))

    designs = []
    for init in DESIGN_INITS + ('published',):
        for variant in VARIANTS:
            members = [r for r in model_rows if r['init'] == init and r['variant'] == variant]
            if not members:
                continue
            designs.append(dict(init=init, variant=variant, models=len(members),
                                accuracy={c: sum(m['accuracy'][c] for m in members) / len(members) for c in CELL_NAMES}))

    attribution = {}
    for c in CELL_NAMES:
        attribution[c] = dict(
            errors=sum(v['cells'][c]['errors'] for v in saved.values()),
            other_memory_value=sum(v['cells'][c]['errors_other_memory_value'] for v in saved.values()),
            position_anchored=sum(v['cells'][c]['errors_position_anchored'] for v in saved.values()))

    z3 = verdicts['Z3_anchored_errors']
    z1 = verdicts['Z1_sanity']
    z4 = verdicts['Z4_query_layout']
    readings = dict(
        standard=f"{z1['models_at_least_95']} of {z1['of']} models score at least 95% on the standard layout.",
        shortcut=f"{verdicts['Z2_position_shortcut']['models_showing']} of {verdicts['Z2_position_shortcut']['of']} "
                 f"models drop at least {SHORTCUT_GAP * 100:.0f} points when one, two or four blank tokens come before the table.",
        anchored=(f"Position-anchored answers: {_pct(z3['anchored_share'], 0)} of the flagged model's wrong answers at two and four "
                  f"blank tokens. Not confirmed as position-anchored: a majority would be needed." if z3['anchored_share'] is not None
                  else 'No wrong answers to attribute.'),
        query_layout={c: f"{v['models_within_10_points']} of {v['of']} models stay within 10 points of standard."
                      for c, v in z4.items()},
    )
    return dict(
        examples_per_cell=512, seed_offset=zlt.SEED_OFFSET, shortcut_gap=SHORTCUT_GAP,
        wall_seconds=results.get('wall_seconds'),
        layouts=[dict(id=c, label=LAYOUT_TEXT[c][0], description=LAYOUT_TEXT[c][1]) for c in CELL_NAMES],
        hypotheses=protocol['hypotheses'],
        verdicts=verdicts,
        readings=readings,
        models=model_rows,
        models_total=len(model_rows),
        models_local=sum(1 for m in model_rows if m['local']),
        shortcut_count=sum(1 for m in model_rows if m['shortcut']),
        designs=designs,
        attribution=attribution,
        provenance=dict(files=dict(verdicts=repo_relative(ZOOLOGY_DIR / 'results.json'),
                                   per_model=repo_relative(ZOOLOGY_DIR / 'cell_results.csv'),
                                   protocol=repo_relative(ZOOLOGY_DIR / 'protocol.json'),
                                   report=repo_relative(ZOOLOGY_DIR / 'REPORT.md'))),
    )


def _kata_summary():
    results = _json(KATA_DIR / 'results.json')
    protocol = _json(KATA_DIR / 'protocol.json')
    verdicts = results['verdicts']
    rows = _csv(KATA_DIR / 'cell_results.csv')
    cells = {}
    for row in rows:
        cells.setdefault(row['cell'], {})[row['model']] = float(row['accuracy'])
    learning_rates = ['lr_0.001', 'lr_0.003', 'lr_0.01']
    selected = verdicts['R1_replication']['selected_learning_rate']
    summary_files = sorted(KATA_OUTPUT.glob('*/lr_*/summary.json'))
    parameters = None
    for path in summary_files:
        data = _json(path)
        if data.get('parameters') is not None:
            parameters = dict(ours=data['parameters'], paper=data.get('paper_parameters'))
            break
    r1 = verdicts['R1_replication']
    r2 = verdicts['R2_rewrite_shortcut']
    shortcut_runs = sum(1 for lr in learning_rates if r2['all_models'][lr]['shortcut'])
    return dict(
        learning_rates=learning_rates,
        selected_learning_rate=selected,
        parameters=parameters,
        replication=dict(
            ours={k.replace('K', ''): v for k, v in r1['ours'].items()},
            paper={k.replace('K=', ''): v for k, v in r1['paper'].items() if k.startswith('K=')},
            ours_average=r1['ours_average'], paper_average=r1['paper']['average'], adequate=r1['adequate']),
        cells={c: {lr: vals.get(lr) for lr in learning_rates} for c, vals in cells.items()},
        cell_names=list(cells.keys()),
        verdicts=verdicts,
        readings=dict(
            replication=f"Average accuracy over sequence lengths K=64 to 384: {_pct(r1['ours_average'], 1)} here, "
                        f"{_pct(r1['paper']['average'], 1)} in the paper.",
            shortcut=f"{shortcut_runs} of {len(learning_rates)} runs show the rewrite shortcut.",
            error_identity=f"Across all three runs' three-pass tests, {verdicts['R3_error_identity']['error_counts'].get('other_key_value', 0)} "
                           f"of {sum(verdicts['R3_error_identity']['error_counts'].values())} wrong answers came from another key's value.",
        ),
        shortcut_runs=shortcut_runs, runs_total=len(learning_rates),
        local_checkpoints=len(list(KATA_OUTPUT.rglob('*.pt'))),
        provenance=dict(files=dict(verdicts=repo_relative(KATA_DIR / 'results.json'),
                                   per_cell=repo_relative(KATA_DIR / 'cell_results.csv'),
                                   protocol=repo_relative(KATA_DIR / 'protocol.json'),
                                   report=repo_relative(KATA_DIR / 'REPORT.md'),
                                   runs=repo_relative(KATA_OUTPUT))),
    )


def summary():
    return dict(zoology=_zoology_summary(), kata=_kata_summary())


# --------------------------------------------------------------------- Zoology try-it

def _slot_of(values, answer):
    """1-based memory slot holding `answer` (values are unique within an example), or None. Shared with shortcut."""
    return shortcut._picked_slot([{'slot': i + 1, 'value': int(v)} for i, v in enumerate(values)], answer)


def _roles(tokens, prefix, query_positions):
    roles = ['filler'] * len(tokens)
    for i in range(PAIRS):
        roles[prefix + 2 * i] = 'key'
        roles[prefix + 2 * i + 1] = 'value'
    for p in query_positions:
        roles[p] = 'query_key'
    return roles


@torch.no_grad()
def evaluate(init, variant, seed, layout, examples, example_seed, show_examples=3):
    """One Zoology model on one layout. The caller holds the model lock. Returns the shortcut-style readout."""
    if layout not in CELL_INDEX:
        raise ControlsError(422, f"layout '{layout}' is not one of the saved Zoology layouts")
    if not 1 <= examples <= 512:
        raise ControlsError(422, 'examples must be 1..512')
    if not 0 <= show_examples <= 8:
        raise ControlsError(422, 'show_examples must be 0..8')
    meta = _find_zoology(init, variant, seed)
    if not meta['local']:
        raise ControlsError(404, f'no local checkpoint for {init}/{variant}/seed {seed}')
    index = CELL_INDEX[layout]
    name, prefix, mode = zlt.CELLS[index]
    cell_seed = example_seed + TRY_SEED_OFFSET + 1000 * index
    tokens, labels, anchored, pair_values = zlt.make_cell(examples, cell_seed, prefix, mode)
    model = _load(str(meta['path']), variant, seed)
    predictions = zlt.evaluate(model, tokens, labels, torch.device('cpu'), BATCH)

    mask = labels != -100
    targets = labels[mask]
    anchor = anchored[mask]
    rows = mask.nonzero()[:, 0].tolist()
    ordinal = (mask.cumsum(1) - 1)[mask].tolist()
    preds = predictions.tolist()
    targets_list = targets.tolist()
    anchor_list = anchor.tolist()

    outcomes, picked, gold_slot_of = [], [], []
    for i, (p, t, a, r) in enumerate(zip(preds, targets_list, anchor_list, rows)):
        values = pair_values[r].tolist()
        gold_slot = _slot_of(values, t)
        pick = _slot_of(values, p)
        if p == t:
            outcome = 'correct'
        elif a != -1 and p == a and a != t:
            outcome = 'position_anchored'
        elif pick is not None:
            outcome = 'other_pair'
        else:
            outcome = 'other'
        outcomes.append(outcome)
        picked.append(pick)
        gold_slot_of.append(gold_slot)

    n = len(preds)
    correct = sum(o == 'correct' for o in outcomes)
    by_query = []
    for q in range(PAIRS):
        idx = [i for i in range(n) if ordinal[i] == q]
        by_query.append(sum(outcomes[i] == 'correct' for i in idx) / len(idx) if idx else None)
    counts = {k: sum(o == k for o in outcomes) for k in ('correct', 'position_anchored', 'other_pair', 'other')}
    sources = {k: v / n for k, v in counts.items()}

    histogram = []
    for slot in range(1, PAIRS + 1):
        hits = [i for i in range(n) if picked[i] == slot]
        from_queried = sum(gold_slot_of[i] == slot for i in hits)
        histogram.append(dict(slot=slot, count=len(hits), from_queried_key=from_queried,
                              from_other_key=len(hits) - from_queried))
    histogram.append(dict(slot=None, count=sum(p is None for p in picked)))

    shown = []
    for r in range(min(show_examples, examples)):
        row_positions = [i for i in range(n) if rows[i] == r]
        query_positions = mask[r].nonzero().flatten().tolist()
        toks = tokens[r].tolist()
        shown.append(dict(
            tokens=toks,
            roles=_roles(toks, prefix, query_positions),
            queries=[dict(position=query_positions[ordinal[i]], gold=targets_list[i], prediction=preds[i],
                          outcome=outcomes[i], picked_slot=picked[i], gold_slot=gold_slot_of[i])
                     for i in row_positions],
        ))

    return dict(
        model=dict(init=init, variant=variant, seed=seed, checkpoint=meta['checkpoint']),
        layout=layout, layout_label=LAYOUT_TEXT[layout][0], layout_description=LAYOUT_TEXT[layout][1],
        examples=examples, example_seed=example_seed, n_answers=n, cell_seed=cell_seed,
        accuracy=correct / n, accuracy_by_query=by_query,
        answer_sources=sources,
        errors=dict(total=n - correct, position_anchored=counts['position_anchored'], other_pair=counts['other_pair'],
                    other=counts['other']),
        slot_histogram=histogram,
        shown=shown,
    )
