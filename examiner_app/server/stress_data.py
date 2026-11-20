"""Stress tests and gate interventions: saved-results summary and live patched checkpoints.

summary() reads the saved result files (nothing is hard-coded). try_run() builds fresh overwrite examples, runs the
local overwrite checkpoints unpatched and with one intervention on the same examples, and records the gate values
(alpha, beta) and recurrent-state norms of one example for both runs. Patches reuse analyze_write_gate.apply_patch
through awg.instrument; the alpha intervention replaces each layer's decay method for the duration of the request.
Every model edit is undone in a finally block, and try_run holds LOCK so patched forwards never overlap.
"""
import csv
import sys
import threading
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_write_gate as awg  # noqa: E402
from recipe_model import RecipeGDN  # noqa: E402
from config import RetrievalConfig  # noqa: E402
from retrieval import RetrievalDataset  # noqa: E402
from checkpoints import repo_relative  # noqa: E402
import shortcut  # noqa: E402  (local overwrite checkpoint discovery, shared with the shortcut page)

RESULTS = ROOT / 'research_results'
STRESS_DIR = RESULTS / 'memory_stress_tests_20261117'
ABLATION_DIR = RESULTS / 'three_seed_gate_ablation_20261117'
STRONGER_DIR = RESULTS / 'stronger_gate_tests_20261117'
MECHANISM_DIR = RESULTS / 'write_gate_mechanism_20261117'
DIAG_DIR = RESULTS / 'three_seed_gate_diagnostics_20261117_final'

# Fresh examples for the live try: no saved study uses this offset (see content_briefs/stress.md).
TRY_SEED_OFFSET = 5_000_000
PATCH_THRESHOLD_POINTS = 2     # "improved" / "worse" means a change of at least this many points
WRITE_NAMES = ('initial', 'first_rewrite', 'second_rewrite', 'third_rewrite')
STALE = frozenset(WRITE_NAMES)
LAYERS = 4
LOCK = threading.Lock()        # one patched or unpatched forward at a time on the shared cached models

INTERVENTIONS = [
    dict(id='none', label='As trained', detail='No change to the trained model.'),
    dict(id='beta_one_latest', label='Full write strength on the newest write',
         detail='Write strength (β) forced to 1 on the newest write, in every layer.'),
    dict(id='beta_zero_rewrite1', label='Block the first replacement',
         detail='Write strength (β) forced to 0 on the first replacement, in every layer.'),
    dict(id='key_align_latest', label='One address for every write of the key',
         detail='Earlier writes of the queried key use the newest write’s key, in every layer.'),
    dict(id='alpha_shared_mean', label='One forgetting rate across channels',
         detail='Decay gate (α) replaced by its average over channels at each token, in every layer. No effect on scalar models.'),
]
PATCH_IDS = {'beta_one_latest', 'beta_zero_rewrite1', 'key_align_latest'}
INTERVENTION_IDS = tuple(i['id'] for i in INTERVENTIONS)


def _csv(path):
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def _num(text):
    return None if text in (None, '') else float(text)


def _init(text):
    """Saved files say 'calibrated' for the mean-decay start; the API uses one label everywhere."""
    return {'calibrated': 'mean-decay'}.get(text, text)


def _round(value, places=4):
    return None if value is None else round(float(value), places)


# ------------------------------------------------------------------------------- saved results

@lru_cache(maxsize=1)
def summary():
    """Everything the stress page shows, parsed from the saved result files."""
    import json
    protocol = json.loads((STRESS_DIR / 'protocol.json').read_text(encoding='utf-8'))
    cells = ([dict(family='mqar', id=name, length=length, keys=keys) for name, length, keys in protocol['mqar_cells']]
             + [dict(family='overwrite', id=name, keys=keys, distance=distance, rewrites=rewrites)
                for name, keys, distance, rewrites in protocol['overwrite_cells']])

    conditions = [dict(family=r['family'], initialization=_init(r['initialization']), variant=r['variant'], cell=r['cell'],
                       accuracy=_round(_num(r['accuracy_mean'])), accuracy_sd=_round(_num(r['accuracy_sd'])),
                       retention=_round(_num(r['retention_mean'])), latest=_round(_num(r['latest_mean'])),
                       stale=_round(_num(r['stale_mean'])))
                  for r in _csv(STRESS_DIR / 'condition_means.csv')]
    per_seed = [dict(family=r['family'], seed=int(r['seed']), initialization=_init(r['initialization']), variant=r['variant'],
                     cell=r['cell'], accuracy=_round(_num(r['exact_query_accuracy'])),
                     latest=_round(_num(r['latest_value_accuracy'])),
                     retention=_round(_num(r['unrelated_association_retention'])),
                     stale=_round(_num(r['stale_value_error_rate'])))
                for r in _csv(STRESS_DIR / 'models.csv')]
    distances = [dict(seed=int(r['seed']), cell=r['cell'], mean=_round(_num(r['mean']), 2),
                      median=_round(_num(r['median']), 2), maximum=int(float(r['maximum'])))
                 for r in _csv(STRESS_DIR / 'mqar_actual_query_distances.csv')]

    shared = [dict(family=r['family'], seed=int(r['seed']), initialization=_init(r['initialization']),
                   baseline=_round(_num(r['baseline_accuracy'])), shared_mean=_round(_num(r['shared_mean_accuracy'])))
              for r in _csv(ABLATION_DIR / 'models.csv')]
    per_model = [dict(family=r['family'], seed=int(r['seed']), initialization=_init(r['initialization']),
                      baseline=_round(_num(r['baseline'])), shared_mean=_round(_num(r['global_mean'])),
                      static=_round(_num(r['global_static'])), shuffle_mean=_round(_num(r['shuffle_mean'])),
                      shuffle_min=_round(_num(r['shuffle_min'])), shuffle_max=_round(_num(r['shuffle_max'])),
                      layer_drops=[_round(_num(r[f'layer_{i}_mean_drop'])) for i in range(1, LAYERS + 1)])
                 for r in _csv(STRONGER_DIR / 'per_model_summary.csv')]
    gate_means = [dict(family=r['family'], initialization=_init(r['initialization']),
                       baseline=_round(_num(r['baseline'])), shared_mean=_round(_num(r['global_mean'])),
                       static=_round(_num(r['global_static'])), shuffle_mean=_round(_num(r['shuffle_mean'])))
                  for r in _csv(STRONGER_DIR / 'condition_means.csv')]

    patch_rows = [r for r in _csv(MECHANISM_DIR / 'patch_results.csv') if r['rewrites'] == '3']
    base = {(r['seed'], r['initialization'], r['variant']): r for r in patch_rows
            if r['patch'] == 'none' and r['patch_layers'] == 'all'}
    patches = []
    for patch in ('beta_one_latest', 'beta_zero_rewrite1', 'key_align_latest', 'beta_copy_rewrite1',
                  'control_beta_one_unrelated'):
        deltas = []
        for r in patch_rows:
            if r['patch'] != patch or r['patch_layers'] != 'all':
                continue
            b = base[(r['seed'], r['initialization'], r['variant'])]
            deltas.append(dict(seed=int(r['seed']), initialization=_init(r['initialization']), variant=r['variant'],
                               latest=float(r['latest_value_accuracy']) - float(b['latest_value_accuracy']),
                               retention=float(r['unrelated_retention']) - float(b['unrelated_retention']),
                               first=float(r['first_rewrite_error_rate']) - float(b['first_rewrite_error_rate'])))
        n = len(deltas)
        mean = lambda key: _round(sum(d[key] for d in deltas) / n, 4)
        patches.append(dict(
            id=patch, models=n, local_models=sum(d['initialization'] != 'joint' for d in deltas),
            latest_change=mean('latest'), retention_change=mean('retention'), first_replacement_change=mean('first'),
            improved=sum(d['latest'] >= PATCH_THRESHOLD_POINTS / 100 for d in deltas),
            worse=sum(d['latest'] <= -PATCH_THRESHOLD_POINTS / 100 for d in deltas),
            per_model=[dict(seed=d['seed'], initialization=_init(d['initialization']), variant=d['variant'],
                            latest=_round(d['latest']), retention=_round(d['retention']))
                       for d in deltas]))
    baseline_values = list(base.values())
    write_baseline = dict(models=len(baseline_values),
                          latest=_round(sum(float(b['latest_value_accuracy']) for b in baseline_values) / len(baseline_values)),
                          retention=_round(sum(float(b['unrelated_retention']) for b in baseline_values) / len(baseline_values)))

    return dict(
        stress=dict(cells=cells, conditions=conditions, per_seed=per_seed, query_distances=distances,
                    examples_per_condition=protocol.get('examples', 256)),
        gate_interventions=dict(shared_mean_per_model=shared, per_model=per_model, condition_means=gate_means),
        write_patches=dict(threshold_points=PATCH_THRESHOLD_POINTS, baseline=write_baseline, patches=patches),
        provenance=dict(files=[repo_relative(p) for p in (
            STRESS_DIR / 'condition_means.csv', STRESS_DIR / 'models.csv', STRESS_DIR / 'mqar_actual_query_distances.csv',
            STRESS_DIR / 'protocol.json', ABLATION_DIR / 'models.csv', STRONGER_DIR / 'per_model_summary.csv',
            STRONGER_DIR / 'condition_means.csv', MECHANISM_DIR / 'patch_results.csv')],
            generated=None),
    )


# ------------------------------------------------------------------------------ live try

def local_models():
    return shortcut.models()


@lru_cache(maxsize=12)
def _model(path_text, variant, seed):
    """Eval-mode RecipeLM, cached separately from the shortcut page's cache so patched state never leaks across pages."""
    return awg.load(Path(path_text), variant, seed)


def _examples(example_seed, distance, rewrites, count):
    """Fresh overwrite examples (answer-free tokens), built the same way as analyze_write_gate.build_cell."""
    config = RetrievalConfig(vocabulary_size=8192, sequence_lengths=(65,), key_counts=(4,),
                             query_distances=(distance,), overwrite_counts=(rewrites,),
                             generator_version='mqar-overwrite-v4')
    dataset = RetrievalDataset(task='overwrite', examples=count, config=config,
                               seed=example_seed + TRY_SEED_OFFSET, split='held_out')
    records = [dataset.example(i) for i in range(count)]
    tokens = torch.stack([r.tokens for r in records]).clone()
    labels = torch.stack([r.labels for r in records])
    for i, record in enumerate(records):          # same rule as stress_recipe_checkpoints.make_overwrite
        for answer in record.answers:
            if answer.label_position + 1 < tokens.shape[1]:
                tokens[i, answer.label_position + 1] = 1
    return records, tokens, labels


def _which_write(prediction, values):
    """Same rule as analyze_write_gate.classify: newest value first, then earlier writes in order."""
    if prediction == values[-1]:
        return 'latest'
    if prediction in values[:-1]:
        return WRITE_NAMES[values[:-1].index(prediction)]
    return 'other'


def _run(model, ctx, records, tokens, labels, positions, record):
    ctx.positions = positions
    ctx.record = [] if record else None
    mask = labels != -100
    flat = model.scored_logits(tokens, mask).argmax(-1)
    grid = torch.full(labels.shape, -1, dtype=torch.long)
    grid[mask] = flat
    layer_records = ctx.record
    ctx.record = None
    answers = []
    for i, rec in enumerate(records):
        values = [int(v) for v in rec.metadata['write_values']]
        latest = next(a.label_position for a in rec.answers if a.kind == 'latest')
        unrelated = next(a.label_position for a in rec.answers if a.kind == 'unrelated')
        pred = int(grid[i, latest])
        pred_u = int(grid[i, unrelated])
        answers.append(dict(prediction=pred, which=_which_write(pred, values), values=values,
                            latest_position=latest, unrelated_position=unrelated,
                            unrelated_gold=int(labels[i, unrelated]), unrelated_prediction=pred_u,
                            unrelated_ok=pred_u == int(labels[i, unrelated])))
    return answers, layer_records


def _rates(answers):
    n = len(answers)
    return dict(
        latest_accuracy=_round(sum(a['which'] == 'latest' for a in answers) / n),
        unrelated_retention=_round(sum(a['unrelated_ok'] for a in answers) / n),
        first_replacement_rate=_round(sum(a['which'] == 'first_rewrite' for a in answers) / n),
        stale_rate=_round(sum(a['which'] in STALE for a in answers) / n),
    )


def _gate_trace(layer_records, index):
    """Per layer: decay gate (mean, min, max over channels), write gate, and Frobenius norm of the recurrent state."""
    layers = []
    for number, rec in enumerate(layer_records, start=1):
        alpha = rec['alpha'][index]                  # [T, gate width]
        beta = rec['beta'][index]                    # [T]
        k, v = rec['k'][index], rec['v'][index]      # [T, key width], [T, value width]
        state = torch.zeros(v.shape[-1], k.shape[-1])
        norms = []
        for t in range(alpha.shape[0]):              # same recurrence as recipe_model.serial_rectangular
            state = state * alpha[t].unsqueeze(0)
            residual = v[t] - state @ k[t]
            state = state + beta[t] * torch.outer(residual, k[t])
            norms.append(float(state.norm()))
        layers.append(dict(layer=number,
                           alpha_mean=[_round(x) for x in alpha.mean(-1).tolist()],
                           alpha_min=[_round(x) for x in alpha.min(-1).values.tolist()],
                           alpha_max=[_round(x) for x in alpha.max(-1).values.tolist()],
                           beta=[_round(x) for x in beta.tolist()],
                           state_norm=[_round(x, 3) for x in norms]))
    return layers


def _roles(record, tokens):
    """Same role rule as app.roles_for (write keys, values, query positions)."""
    md = record.metadata
    roles = ['filler'] * len(tokens)
    for p in md['all_write_positions']:
        roles[p] = 'value'
        roles[p - 1] = 'key'
    for p in md['write_positions']:
        roles[p] = 'target_value'
        roles[p - 1] = 'target_key'
    for a in record.answers:
        roles[a.label_position] = 'query_latest' if a.kind == 'latest' else 'query_unrelated'
        if a.label_position - 1 >= 0 and tokens[a.label_position - 1] == 0:
            roles[a.label_position - 1] = 'marker'
    return roles


def _set_intervention(model, intervention, ctx):
    ctx.patch = intervention if intervention in PATCH_IDS else 'none'
    ctx.layers = set(range(LAYERS)) if intervention in PATCH_IDS else set()
    if intervention == 'alpha_shared_mean':
        for layer in _gdn_layers(model):
            original = layer.decay

            def shared(x, _original=original):
                alpha = _original(x)
                return alpha.mean(-1, keepdim=True).expand_as(alpha)
            layer.decay = shared


def _clear_intervention(model, ctx):
    ctx.patch, ctx.layers = 'none', set()
    for layer in _gdn_layers(model):
        layer.__dict__.pop('decay', None)


def _gdn_layers(model):

    return [m for m in model.modules() if isinstance(m, RecipeGDN)]


def try_run(init, variant, seed, intervention, rewrites, distance, count, example_seed, show):
    """Runs the same fresh examples unpatched and with the intervention. Raises ValueError for unbuildable settings."""
    checkpoint = shortcut.local_checkpoints().get((init, variant, seed))
    if checkpoint is None:
        raise FileNotFoundError(f'No local checkpoint for {init}/{variant}/seed {seed}')
    try:
        records, tokens, labels = _examples(example_seed, distance, rewrites, count)
    except (ValueError, AssertionError) as exc:
        raise ValueError(f'could not build these examples: {exc}')
    positions = awg.Positions(records)

    with LOCK, torch.no_grad():
        model = _model(str(checkpoint), variant, seed)
        ctx = awg.instrument(model)
        try:
            before, before_layers = _run(model, ctx, records, tokens, labels, positions, record=True)
            _set_intervention(model, intervention, ctx)
            after, after_layers = _run(model, ctx, records, tokens, labels, positions, record=True)
        finally:
            _clear_intervention(model, ctx)
            awg.restore(model)
        parameters = sum(p.numel() for p in model.parameters())

    changed = sum(b['prediction'] != a['prediction'] for b, a in zip(before, after))
    fixed = sum(b['which'] != 'latest' and a['which'] == 'latest' for b, a in zip(before, after))
    broken = sum(b['which'] == 'latest' and a['which'] != 'latest' for b, a in zip(before, after))

    rec = records[show]
    row = show
    shown = tokens[row].tolist()
    writes = []
    targets = set(rec.metadata['write_positions'])
    for slot, p in enumerate(sorted(rec.metadata['all_write_positions']), start=1):
        writes.append(dict(slot=slot, key=int(shown[p - 1]), value=int(shown[p]), is_target=p in targets))

    def picked(prediction):
        slots = [w['slot'] for w in writes if w['value'] == prediction]
        return max(slots) if slots else None

    queries = []
    for answer in rec.answers:
        if answer.kind == 'latest':
            b, a = before[row], after[row]
            queries.append(dict(kind='latest', gold=int(rec.metadata['write_values'][-1]),
                                before=dict(prediction=b['prediction'], which=b['which'], picked_slot=picked(b['prediction'])),
                                after=dict(prediction=a['prediction'], which=a['which'], picked_slot=picked(a['prediction']))))
        else:
            queries.append(dict(kind='unrelated', gold=int(labels[row, answer.label_position]),
                                before=dict(prediction=before[row]['unrelated_prediction'], ok=before[row]['unrelated_ok']),
                                after=dict(prediction=after[row]['unrelated_prediction'], ok=after[row]['unrelated_ok'])))

    return dict(
        model=dict(init=init, variant=variant, seed=seed, parameters=parameters,
                   checkpoint=repo_relative(checkpoint)),
        setup=dict(intervention=intervention, rewrites=rewrites, distance=distance, examples=count,
                   example_seed=example_seed, show=show, seed_offset=TRY_SEED_OFFSET,
                   no_effect_expected=(intervention == 'alpha_shared_mean' and variant == 'scalar')),
        before=_rates(before), after=_rates(after),
        changed_answers=changed, fixed=fixed, broken=broken,
        example=dict(tokens=[int(t) for t in shown], roles=_roles(rec, shown), writes=writes, queries=queries,
                     values=[int(v) for v in rec.metadata['write_values']]),
        gates=dict(before=_gate_trace(before_layers, row), after=_gate_trace(after_layers, row)),
    )
