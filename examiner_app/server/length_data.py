"""Longer sequences and filler decay: saved-results summaries and live MQAR runs (read-only).

Saved numbers are parsed at request time from research_results/mqar_length_extension_20261119 and
research_results/filler_decay_diagnostic_20261119. Nothing is hard-coded. run() feeds the local MQAR checkpoints
extra token-0 filler after the 8-token prefix, the same insertion as run_paper_mqar_length_extension.py, and records
the decay gate (alpha) on the inserted span. The caller (routers/length.py) holds the lock around run().
"""
import json
import math
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyze_write_gate as awg  # noqa: E402
from recipe_data import audit_mqar, mqar, manifest, RecipeData  # noqa: E402
from recipe_model import RecipeGDN  # noqa: E402
from checkpoints import repo_relative  # noqa: E402

SWEEP_DIR = ROOT / 'research_results' / 'mqar_length_extension_20261119'
DECAY_DIR = ROOT / 'research_results' / 'filler_decay_diagnostic_20261119'
MODEL_ROOT = ROOT / 'output' / 'calibrated_recipe_study' / 'matched-decay-v1'

PREFIX = 8             # the four key/value writes come first
TRAIN_LENGTH = 64
LENGTHS = (64, 256, 1024)
OFFSET = 1950000       # evaluation seed range (sweep and diagnostic); training used 100000 + seed
SEEDS = (42, 43, 44)
VARIANTS = ('scalar', 'channel-wise')
FOLDER_TO_INIT = {'default': 'default', 'calibrated': 'mean-decay'}   # folder label -> page label
SAVED_TO_INIT = {'default': 'default', 'calibrated': 'mean-decay'}    # saved-json label -> page label
MAX_FILLER = 960       # 1,024 - 64
MAX_EXAMPLES = 16      # one 16 x 1,024 forward takes about 5 to 7 s on CPU
LOG_FLOOR = -300.0     # log10 floor for products that underflow

_lock = threading.Lock()


class LengthError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


# ------------------------------------------------------------------------------ local models

@lru_cache(maxsize=1)
def _local_checkpoints():
    """{(init_label, variant, seed): checkpoint Path} for the MQAR runs present on disk."""
    found = {}
    for path in sorted(MODEL_ROOT.glob('*/mqar/seed_*/*/*/latest_checkpoint.pt')):
        session, family, seed_dir, folder, variant = path.relative_to(MODEL_ROOT).parts[:5]
        seed = int(seed_dir.split('_', 1)[1])
        if folder not in FOLDER_TO_INIT or variant not in VARIANTS or seed not in SEEDS:
            continue
        key = (FOLDER_TO_INIT[folder], variant, seed)
        if key in found:
            raise RuntimeError(f'Ambiguous MQAR checkpoints for {key}: {found[key]} and {path}')
        found[key] = path
    return found


def models():
    """The local MQAR checkpoints, in (init, variant, seed) order."""
    return [dict(init=init, variant=variant, seed=seed, checkpoint=repo_relative(path))
            for (init, variant, seed), path in sorted(_local_checkpoints().items())]


def _checkpoint(init, variant, seed):
    path = _local_checkpoints().get((init, variant, seed))
    if path is None:
        raise LengthError(404, f'No MQAR checkpoint on this machine for {init}/{variant}/seed {seed}.')
    return path


@lru_cache(maxsize=12)
def _load(path_text, variant, seed):
    """Eval-mode RecipeLM on CPU, cached (at most 12 resident). Shared: never left modified."""
    return awg.load(Path(path_text), variant, seed)


# ----------------------------------------------------------------------------- saved summary

def _sweep_rows():
    return _json(SWEEP_DIR / 'results.json')


def _cell(row, local):
    init = SAVED_TO_INIT[row['initialization']]
    key = (init, row['variant'], row['seed'])
    return dict(init=init, variant=row['variant'], seed=row['seed'], length=row['length'],
                correct=row['correct'], total=row['total'], accuracy=row['exact_query_accuracy'],
                checkpoint_sha256=row['checkpoint_sha256'], local=key in local)


def summary():
    sweep = _sweep_rows()
    verified = _json(SWEEP_DIR / 'verified_results.json')
    if sweep.get('status') != 'complete' or verified.get('status') != 'complete':
        raise LengthError(503, 'The saved length sweep is not complete.')
    local = set(_local_checkpoints())
    cells = [_cell(r, local) for r in sweep['results']]

    # Cross-check: the two saved sweep files must agree cell for cell.
    def key(r):
        return (r['seed'], r['initialization'], r['variant'], r['length'])
    other = {key(r): r for r in verified['results']}
    agree = sum(1 for r in sweep['results']
                if key(r) in other and other[key(r)]['correct'] == r['correct']
                and other[key(r)]['checkpoint_sha256'] == r['checkpoint_sha256'])

    by_model = []
    for init, variant, seed in sorted({(c['init'], c['variant'], c['seed']) for c in cells}):
        accuracy = {}
        for c in cells:
            if (c['init'], c['variant'], c['seed']) == (init, variant, seed):
                accuracy[str(c['length'])] = c['accuracy']
        by_model.append(dict(init=init, variant=variant, seed=seed, local=(init, variant, seed) in local,
                             accuracy=accuracy))

    means = []
    for length in LENGTHS:
        values = [c['accuracy'] for c in cells if c['length'] == length]
        means.append(dict(length=length, mean=float(np.mean(values)), min=float(min(values)),
                          max=float(max(values)), models=len(values)))

    replay_report = _json(SWEEP_DIR / 'targeted_replays' / 'targeted_replay_report.json')
    replays = []
    for path in sorted((SWEEP_DIR / 'targeted_replays').glob('*_summary.json')):
        body = _json(path)
        init = SAVED_TO_INIT[body['initialization']]
        variant = body['variant']
        seed = int(body['data_manifest']['seed']) - OFFSET
        length = TRAIN_LENGTH + int(body['filler_protocol']['inserted_filler_count'])
        match = next((c for c in cells if (c['init'], c['variant'], c['seed'], c['length'])
                      == (init, variant, seed, length)), None)
        replays.append(dict(init=init, variant=variant, seed=seed, length=length,
                            correct=body['correct'], answers=body['answers'], accuracy=body['accuracy'],
                            sweep_correct=None if match is None else match['correct'],
                            matches_sweep=None if match is None else match['correct'] == body['correct'],
                            checkpoint_sha256=body['checkpoint_sha256']))
    hash_audit = _json(SWEEP_DIR / 'checkpoint_hash_audit.json')

    return dict(
        lengths=list(LENGTHS),
        training_length=TRAIN_LENGTH,
        prefix_tokens=PREFIX,
        examples_per_seed=_json(SWEEP_DIR / 'protocol.json')['examples_per_seed'],
        cells=cells,
        models=by_model,
        means=means,
        replays=replays,
        replay_status=replay_report.get('status'),
        verification=dict(
            cells=len(cells),
            verified_cells_agree=agree,
            checkpoint_hashes=hash_audit.get('matches'),
            checkpoint_audit_status=hash_audit.get('status'),
            local_checkpoints=len(local),
        ),
        protocol=_json(SWEEP_DIR / 'protocol.json'),
        provenance=dict(
            sweep=repo_relative(SWEEP_DIR / 'results.json'),
            verified=repo_relative(SWEEP_DIR / 'verified_results.json'),
            audit=repo_relative(SWEEP_DIR / 'checkpoint_hash_audit.json'),
            replays=repo_relative(SWEEP_DIR / 'targeted_replays' / 'targeted_replay_report.json'),
        ),
    )


@lru_cache(maxsize=1)
def _decay_rows():
    rows = _json(DECAY_DIR / 'results.json')
    if rows.get('status') != 'complete':
        raise LengthError(503, 'The saved filler decay diagnostic is not complete.')
    out = []
    for row in rows['results']:
        init = SAVED_TO_INIT[row['initialization']]
        m = row['metrics']
        layers = []
        for i, L in enumerate(row['layers']):
            ln10 = math.log(10)
            layers.append(dict(
                layer=i + 1,
                alpha_p50=L['alpha_p50'],
                half_life_tokens=None if math.isinf(L['effective_half_life_tokens']) else L['effective_half_life_tokens'],
                product_p10=L['product_p10'], product_p50=L['product_p50'], product_p90=L['product_p90'],
                log10_product_p10=max(L['log_product_p10'] / ln10, LOG_FLOOR),
                log10_product_p50=max(L['log_product_p50'] / ln10, LOG_FLOOR),
                log10_product_p90=max(L['log_product_p90'] / ln10, LOG_FLOOR),
                channels=L['gate_channels'], examples=L['examples']))
        out.append(dict(init=init, variant=row['variant'], seed=row['seed'], length=row['length'],
                        filler_tokens=row['filler_stop'] - row['filler_start'],
                        correct=m['correct'], total=m['total'], layers=layers))
    return out


def decay():
    out = _decay_rows()
    hist = _json(DECAY_DIR / 'product_histogram.json')
    histogram = dict(
        length=hist['length'], layer=hist['layer'], examples_per_seed=hist['examples_per_seed'],
        initializer=hist['initializer'], inserted_tokens=hist['inserted_tokens'],
        records=[dict(seed=r['seed'], correct=r['metrics']['correct'], total=r['metrics']['total'],
                      bin_edges=r['bin_edges'], counts=r['counts']) for r in hist['records']])
    protocol = _json(DECAY_DIR / 'protocol.json')
    return dict(
        rows=out,
        histogram=histogram,
        protocol=dict(seeds=protocol['seeds'], lengths=protocol['lengths'], examples_per_seed=protocol['examples_per_seed'],
                      prefix_tokens_before_filler=protocol['prefix_tokens_before_filler'],
                      filler_token=protocol['filler_token'], alpha_definition=protocol['alpha_definition'],
                      interpretation_guardrail=protocol['interpretation_guardrail']),
        provenance=dict(results=repo_relative(DECAY_DIR / 'results.json'),
                        histogram=repo_relative(DECAY_DIR / 'product_histogram.json'),
                        protocol=repo_relative(DECAY_DIR / 'protocol.json')),
    )


# ----------------------------------------------------------------------------- live run

def _extend(tokens, targets, filler):
    """Insert `filler` token-0 positions after the prefix; labels get -100 there (as the sweep script does)."""
    n = len(tokens)
    extended = torch.cat((tokens[:, :PREFIX], tokens.new_zeros((n, filler)), tokens[:, PREFIX:]), dim=1)
    labels = torch.cat((targets[:, :PREFIX], targets.new_full((n, filler), -100), targets[:, PREFIX:]), dim=1)
    return extended, labels


def run(init, variant, seed, filler, examples):
    """Run one checkpoint on fresh MQAR examples with `filler` extra tokens.

    Returns accuracy at this filler count, the decay-only survival curve for every filler count 0..filler (causal:
    one pass with `filler` tokens gives every shorter count exactly), and the first example's tokens and answers.
    """
    if not (0 <= filler <= MAX_FILLER):
        raise LengthError(422, f'filler must be between 0 and {MAX_FILLER}.')
    if not (1 <= examples <= MAX_EXAMPLES):
        raise LengthError(422, f'examples must be between 1 and {MAX_EXAMPLES}.')
    path = _checkpoint(init, variant, seed)
    tokens, targets = mqar(examples, length=TRAIN_LENGTH, seed=OFFSET + seed)
    extended, labels = _extend(tokens, targets, filler)
    audit = audit_mqar(extended, labels)
    if not audit['passed']:
        raise LengthError(500, 'The generated MQAR examples failed the label audit.')
    mask = labels != -100

    with _lock, torch.no_grad():
        model = _load(str(path), variant, seed)
        layers = list(model.network.backbone.layers)
        captured = [[] for _ in layers]
        hooks = []
        try:
            for index, layer in enumerate(layers):
                mixer = layer.sequence_mixer
                if not isinstance(mixer, RecipeGDN):
                    raise LengthError(500, f'Unexpected sequence mixer at layer {index + 1}.')

                def capture(module, args, store=captured[index]):
                    store.append(module.decay(args[0])[:, PREFIX:PREFIX + filler].detach().double().cpu())
                hooks.append(mixer.register_forward_pre_hook(capture))
            logits = model.scored_logits(extended, mask)
        finally:
            for hook in hooks:
                hook.remove()

    predictions = logits.argmax(-1)
    gold = labels[mask]
    correct_flags = (predictions == gold).tolist()
    total = len(correct_flags)
    correct = int(sum(correct_flags))
    positions = mask.nonzero()                      # row-major, matching scored_logits
    per_example = [[] for _ in range(examples)]
    for idx, (row, col) in enumerate(positions.tolist()):
        per_example[row].append(dict(position=col, gold=int(gold[idx]), prediction=int(predictions[idx]),
                                     correct=bool(correct_flags[idx])))

    survival = []
    at_filler = []
    for index, chunks in enumerate(captured):
        if filler == 0:
            survival.append(dict(layer=index + 1, log10_p10=[0.0], log10_p50=[0.0], log10_p90=[0.0]))
            at_filler.append(dict(layer=index + 1, product_p10=1.0, product_p50=1.0, product_p90=1.0,
                                  log10_p50=0.0, channels=1))
            continue
        alpha = chunks[0].reshape(examples, filler, -1)            # [examples, filler, channels]
        log_alpha = alpha.clamp_min(torch.finfo(torch.float64).tiny).log()
        cumulative = log_alpha.cumsum(dim=1).numpy()                 # log product over the first n filler tokens
        channels = cumulative.shape[2]
        # Percentiles run over examples x channels, as in the saved diagnostic (linear products, then log10).
        products = np.exp(np.clip(cumulative, -745.0, 0.0)).transpose(1, 0, 2).reshape(filler, examples * channels)
        p = {q: np.percentile(products, q, axis=1) for q in (10, 50, 90)}
        with np.errstate(divide='ignore'):
            log10 = {q: np.maximum(np.log10(np.maximum(p[q], 0.0)), LOG_FLOOR) for q in (10, 50, 90)}
        survival.append(dict(
            layer=index + 1,
            log10_p10=[0.0] + [round(float(x), 4) for x in log10[10]],
            log10_p50=[0.0] + [round(float(x), 4) for x in log10[50]],
            log10_p90=[0.0] + [round(float(x), 4) for x in log10[90]]))
        at_filler.append(dict(layer=index + 1,
                              product_p10=float(p[10][-1]), product_p50=float(p[50][-1]), product_p90=float(p[90][-1]),
                              log10_p50=float(log10[50][-1]), channels=channels))

    first = per_example[0]
    first_tokens = extended[0].tolist()
    return dict(
        model=dict(init=init, variant=variant, seed=seed, checkpoint=repo_relative(path),
                   parameters=sum(p.numel() for p in model.parameters())),
        filler=filler,
        length=TRAIN_LENGTH + filler,
        examples=examples,
        data_seed=OFFSET + seed,
        accuracy=correct / total,
        correct=correct,
        total=total,
        audit_passed=bool(audit['passed']),
        example=dict(tokens=[int(t) for t in first_tokens], queries=first,
                     correct=sum(q['correct'] for q in first), total=len(first)),
        survival=dict(steps=list(range(filler + 1)), layers=survival),
        at_filler=at_filler,
        saved=saved_points(init, variant, seed, filler),
    )


def saved_points(init, variant, seed, filler):
    """Saved diagnostic endpoints for this model at the same filler count (for comparison on the page), or None."""
    for row in _decay_rows():
        if (row['init'], row['variant'], row['seed'], row['filler_tokens']) == (init, variant, seed, filler):
            return dict(correct=row['correct'], total=row['total'],
                        layers=[dict(layer=L['layer'], log10_product_p50=L['log10_product_p50']) for L in row['layers']])
    return None
