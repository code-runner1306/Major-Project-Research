"""Selective reset pilot: saved results and the one local checkpoint.

Reads output/selective_reset/*.json at request time and runs the checkpoint on CPU. Trains nothing and writes nothing.
Task constants come from selective_reset_pilot.py (imported, not copied).
"""
import json
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import selective_reset_pilot as pilot  # noqa: E402  (puts third_party/dualgoose on sys.path)
from dualgoose.model import build_tiny_denoiser  # noqa: E402

RESULT_DIR = ROOT / 'output' / 'selective_reset'
# The file name of the channel-wise seed-42 run has a capital D in the repo; keep the real name.
RUN_FILES = {
    ('gated_deltanet', 42): 'gated_Deltanet_42.json',
    ('gated_deltanet', 43): 'gated_deltanet_43.json',
    ('gated_deltanet_scalar', 42): 'gated_deltanet_scalar_42.json',
    ('gated_deltanet_scalar', 43): 'gated_deltanet_scalar_43.json',
}
CHECKPOINT_FILES = {
    ('gated_deltanet', 42): ('gated_deltanet_42.pt', 'gated_deltanet_42.zip'),
}
DECAY_NAME = {'gated_deltanet': 'channel-wise', 'gated_deltanet_scalar': 'scalar'}
QUERY_NAMES = {'target': 'Cleared key', 'survivor': 'Other workspace key', 'rewrite': 'Rewritten key',
               'absent': 'Missing key'}
METRICS = ['reset_target', 'noop_target', 'reset_survivor', 'noop_survivor', 'reset_rewrite', 'noop_rewrite',
           'reset_absent', 'noop_absent', 'reset_joint', 'noop_joint', 'stale_value_returned']
# Same arguments as train() in selective_reset_pilot.py (the JSON files do not store the model shape).
MODEL_KWARGS = dict(d_model=32, n_layers=2, n_heads=1, short_conv=4, direction='forward', merge='static')
VALUE_COUNT = 70 - 38  # value tokens 38..69, read from the pilot's episodes()
_lock = threading.Lock()  # one forward pass at a time; the cached model is shared


def _read_run(family, seed):
    data = json.loads((RESULT_DIR / RUN_FILES[(family, seed)]).read_text(encoding='utf-8'))
    return data


def _checkpoint_path(family, seed):
    for name in CHECKPOINT_FILES.get((family, seed), ()):
        path = RESULT_DIR / name
        if path.exists():
            return path
    return None


def is_live():
    return _checkpoint_path('gated_deltanet', 42) is not None


def summary():
    runs = []
    for (family, seed) in RUN_FILES:
        data = _read_run(family, seed)
        assert data['family'] == family and data['seed'] == seed, RUN_FILES[(family, seed)]
        by_load = [{'load': int(load), **{m: acc[m] for m in METRICS}} for load, acc in sorted(
            data['accuracy'].items(), key=lambda item: int(item[0]))]
        runs.append({
            'family': family,
            'decay': DECAY_NAME[family],
            'seed': seed,
            'parameters': data['parameters'],
            'training_seconds': round(data['training_seconds'], 1),
            'steps': data['settings']['steps'],
            'batch': data['settings']['batch'],
            'by_load': by_load,
            'has_checkpoint': _checkpoint_path(family, seed) is not None,
        })

    cells = [row for run in runs for row in run['by_load']]

    def mean(metric, load=None):
        values = [row[metric] for run in runs for row in run['by_load'] if load is None or row['load'] == load]
        return sum(values) / len(values)

    per_query = {m: mean(m) for m in METRICS}
    by_load = [{'load': load, **{m: mean(m, load) for m in METRICS}} for load in pilot.LOADS]
    checkpoints = [{'decay': DECAY_NAME[f], 'seed': s, 'file': _checkpoint_path(f, s).name}
                   for (f, s) in CHECKPOINT_FILES if _checkpoint_path(f, s) is not None]
    return {
        'task': {
            'memory_loads': list(pilot.LOADS),
            'keys_per_workspace': 16,
            'queries': list(pilot.QUERIES),
            'query_names': QUERY_NAMES,
            'values': VALUE_COUNT,
            'guess_rate': 1 / VALUE_COUNT,
            'length': pilot.LENGTH,
            'steps': runs[0]['steps'],
            'batch': runs[0]['batch'],
        },
        'runs': runs,
        'per_query': per_query,
        'by_load': by_load,
        'max_joint': max(max(row['reset_joint'], row['noop_joint']) for row in cells),
        'max_survivor': max(max(row['reset_survivor'], row['noop_survivor']) for row in cells),
        'min_reset_target': min(row['reset_target'] for row in cells),
        'max_noop_target': max(row['noop_target'] for row in cells),
        'live': bool(checkpoints),
        'checkpoints': checkpoints,
        'provenance': {
            'results': [f'output/selective_reset/{name}' for name in RUN_FILES.values()],
            'task': 'selective_reset_pilot.py',
            'checkpoint': [f'output/selective_reset/{p.name}' for p in
                           (_checkpoint_path(f, s) for (f, s) in CHECKPOINT_FILES) if p is not None],
        },
    }


@lru_cache(maxsize=4)
def load_model(family, seed):
    path = _checkpoint_path(family, seed)
    if path is None:
        raise FileNotFoundError('no checkpoint for this run on this machine')
    model = build_tiny_denoiser(pilot.VOCAB, pilot.LENGTH, model_type=family, **MODEL_KWARGS)
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def live_model():
    """The one checkpoint the page can run (channel-wise, seed 42). Raises if it is absent."""
    return load_model('gated_deltanet', 42)


def _predict(model, x):
    with torch.no_grad():
        return model(torch.from_numpy(x), torch.ones(len(x)))[:, -1].softmax(-1)


def token_role(token):
    if token == 0:
        return 'filler'
    if token == 1:
        return 'noop'
    if token in (2, 3):
        return 'reset'
    if token in (4, 5):
        return 'workspace'
    if 6 <= token <= 37:
        return 'key'
    if 38 <= token <= 69:
        return 'value'
    return 'answer'


def token_label(token):
    role = token_role(token)
    if role == 'value':
        return f'value {token - 37}'
    if role == 'key':
        return f'key {token - 5}'
    if role == 'workspace':
        return 'red' if token == 4 else 'blue'
    if role == 'reset':
        return 'reset red' if token == 2 else 'reset blue'
    if role == 'noop':
        return 'no-op'
    if role == 'answer':
        return 'empty'
    return ''


def try_example(load, reset, query, seed):
    """One fresh episode from the pilot's own generator, answered by the local checkpoint."""
    kind = pilot.QUERIES.index(query)
    rng = np.random.default_rng(seed)
    ep = pilot.episodes(rng, 1, load)
    x, y = pilot.branch(ep, reset, kind)
    stale, survivor_value = int(ep[-1][0][0]), int(ep[-1][1][0])
    with _lock:
        probs = _predict(live_model(), x)[0]
    top = torch.topk(probs, 3)
    prediction = int(top.indices[0])
    tokens = x[0].tolist()
    return {
        'load': load,
        'reset': reset,
        'query': query,
        'seed': seed,
        'tokens': [{'id': t, 'role': token_role(t), 'label': token_label(t)} for t in tokens],
        'expected': int(y[0]),
        'expected_label': token_label(int(y[0])),
        'prediction': prediction,
        'prediction_label': token_label(prediction),
        'correct': prediction == int(y[0]),
        'top3': [{'token': int(i), 'label': token_label(int(i)), 'prob': round(float(p), 4)}
                 for p, i in zip(top.values, top.indices)],
        'cleared_key_old_value': stale,
        'survivor_value': survivor_value,
        'model': {'decay': 'channel-wise', 'seed': 42, 'parameters': sum(p.numel() for p in live_model().parameters())},
    }


def _answer_kind(pred, expected, stale, survivor_value):
    if pred == expected:
        return 'correct'
    if pred == pilot.EMPTY:
        return 'empty'
    if pred == stale:
        return 'old value of cleared key'
    if pred == survivor_value:
        return "other workspace's value"
    return 'something else'


KINDS = ['correct', 'empty', 'old value of cleared key', "other workspace's value", 'something else']


def profile(per_load=256, seed=20261120):
    """What the live checkpoint answers, over fixed-seed episodes: a share per answer kind, per query, reset or no-op."""
    model = live_model()
    rng = np.random.default_rng(seed)
    out = {}
    with _lock:
        for load in pilot.LOADS:
            ep = pilot.episodes(rng, per_load, load)
            stale, survivor_value = ep[-1][0], ep[-1][1]
            for reset in (True, False):
                for kind, name in enumerate(pilot.QUERIES):
                    x, y = pilot.branch(ep, reset, kind)
                    pred = _predict(model, x).argmax(-1).numpy()
                    counts = {k: 0 for k in KINDS}
                    for p_, e_, s_, v_ in zip(pred, y, stale, survivor_value):
                        counts[_answer_kind(int(p_), int(e_), int(s_), int(v_))] += 1
                    key = f'{"reset" if reset else "noop"}_{name}'
                    out.setdefault(str(load), {})[key] = {k: round(v / per_load, 4) for k, v in counts.items()}
    return {
        'episodes_per_cell': per_load,
        'kinds': KINDS,
        'checkpoint': {'decay': 'channel-wise', 'seed': 42},
        'by_load': out,
    }
