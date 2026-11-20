"""Decay initialization page: saved results, saved gate values, and a live try-it on the 12 local overwrite models.

Endpoints (all JSON, all read-only with respect to research code and result files):
  GET  /api/decay-init/summary   learning-speed numbers for default, mean-decay, and joint starts; fresh-example
                                 numbers for default and mean-decay, from the saved study CSVs
  GET  /api/decay-init/gates     per-layer decay values at start and after training (server/data/decay_init_gates.json,
                                 written by server/decay_init_build.py from the checkpoints)
  GET  /api/decay-init/models    the 12 local overwrite models (shortcut.models())
  POST /api/decay-init/try       one fresh overwrite example run through a local checkpoint on CPU, with the decay
                                 values of every layer and token

The try endpoint shares the server's model lock (app._lock) and the model cache of the shortcut page, and it never
patches a model.
"""
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

SERVER = Path(__file__).resolve().parents[1]
ROOT = SERVER.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

import confirm_block_layout as cbl  # noqa: E402
import internals  # noqa: E402
import shortcut  # noqa: E402
from checkpoints import repo_relative  # noqa: E402

# Fresh evaluation data: the seed offset app.py and the shortcut page use.
cbl.SEED_OFFSET = 3_000_000

router = APIRouter(prefix='/api/decay-init')

RESULTS = ROOT / 'research_results'
EXPLORATION = RESULTS / 'three_seed_exploration_20261117' / 'models.csv'
CONFIRMATION = RESULTS / 'three_seed_confirmation_20261117' / 'models.csv'
JOINT_RESULTS = RESULTS / 'completed_joint_decay_review_20261117' / 'training_comparison.csv'
GATES_FILE = SERVER / 'data' / 'decay_init_gates.json'
WRITE_NAMES = ('initial', 'first_rewrite', 'second_rewrite', 'third_rewrite')
SEEDS = (42, 43, 44)
INITS = ('default', 'mean-decay', 'joint')
VARIANTS = ('scalar', 'channel-wise')
FAMILIES = ('overwrite', 'mqar')


def _read_csv(path):
    return list(csv.DictReader(path.read_text(encoding='utf-8').splitlines()))


def _init(label):
    """CSV label (default / calibrated) -> API label (default / mean-decay)."""
    return shortcut.FOLDER_TO_INIT.get(label, label)


def _sample_sd(values):
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return (sum((v - mean) ** 2 for v in values) / (n - 1)) ** 0.5


def _measure(rows, family, column, value_of, inits=INITS):
    """Rows for one measure: one entry per (init, variant) with the three seed values and their mean and SD."""
    table = {}
    for row in rows:
        if row['family'] != family:
            continue
        key = (_init(row['initialization']), row['variant'], int(row['seed']))
        table[key] = value_of(row)
    out = []
    for init in inits:
        for variant in VARIANTS:
            seeds = [table[(init, variant, s)] for s in SEEDS]
            out.append(dict(init=init, variant=variant, seeds=seeds,
                            mean=sum(seeds) / len(seeds), sd=_sample_sd(seeds)))
    return out


def _contrast(rows, inits=INITS):
    """Channel-wise minus scalar for each initialization; mean-decay interaction remains versus default."""
    by = {(r['init'], r['variant']): r['seeds'] for r in rows}

    def block(values):
        return dict(seeds=values, mean=sum(values) / len(values), sd=_sample_sd(values))

    gaps = {}
    for init in inits:
        values = [c - scalar for c, scalar in zip(by[(init, 'channel-wise')], by[(init, 'scalar')])]
        key = 'mean_decay' if init == 'mean-decay' else init
        gaps[key] = block(values)
    if 'default' in inits and 'mean-decay' in inits:
        default_gap = gaps['default']['seeds']
        mean_gap = gaps['mean_decay']['seeds']
        gaps['interaction'] = block([mean - default for mean, default in zip(mean_gap, default_gap)])
    return gaps


@lru_cache(maxsize=1)
def summary_data():
    exploration = _read_csv(EXPLORATION)
    confirmation = _read_csv(CONFIRMATION)
    joint_learning = []
    for row in _read_csv(JOINT_RESULTS):
        value = row['retention_aulc'] if row['family'] == 'overwrite' else row['accuracy_aulc']
        joint_learning.append(dict(row, primary_aulc=value))
    learning_rows = exploration + joint_learning
    families = {}
    for family in FAMILIES:
        learning = _measure(learning_rows, family, 'primary_aulc', lambda r: float(r['primary_aulc']))
        fresh = _measure(confirmation, family, 'accuracy', lambda r: float(r['accuracy']), inits=INITS[:2])
        measures = dict(learning=learning, fresh=fresh)
        contrasts = dict(learning=_contrast(learning), fresh=_contrast(fresh))
        if family == 'overwrite':
            measures['fresh_retention'] = _measure(confirmation, family, 'retention', lambda r: float(r['retention']), inits=INITS[:2])
            contrasts['fresh_retention'] = _contrast(measures['fresh_retention'])
        families[family] = dict(measures=measures, contrasts=contrasts)
    return dict(
        seeds=list(SEEDS),
        families=families,
        definitions=dict(
            learning='AULC over epochs 1 to 16 (0 to 1 scale). Overwrite: unrelated-association retention. MQAR: answer-free query accuracy.',
            fresh='Final accuracy on 512 fresh examples per seed for default and mean-decay only. Overwrite: overall (latest and unrelated answers). MQAR: query accuracy.',
            fresh_retention='Final unrelated-association retention on the same fresh examples (overwrite only).',
            sd='Sample standard deviation across the three seeds.',
        ),
        provenance=dict(files=[repo_relative(EXPLORATION), repo_relative(CONFIRMATION), repo_relative(JOINT_RESULTS)],
                        note='Joint AULC values come from the separate completed joint study; fresh-example endpoints remain default/mean-decay only.'),
    )


@router.get('/summary')
def summary():
    return summary_data()


@lru_cache(maxsize=1)
def _gates():
    if not GATES_FILE.exists():
        raise FileNotFoundError(GATES_FILE.name)
    return json.loads(GATES_FILE.read_text(encoding='utf-8'))


@router.get('/gates')
def gates():
    try:
        data = _gates()
    except FileNotFoundError:
        raise HTTPException(404, 'server/data/decay_init_gates.json is missing; run decay_init_build.py once')
    return dict(data, provenance=dict(file=repo_relative(GATES_FILE)))


@router.get('/models')
def models():
    return shortcut.models()


class TryRequest(BaseModel):
    init: Literal['default', 'mean-decay']
    variant: Literal['scalar', 'channel-wise']
    seed: Literal[42, 43, 44]
    layout: str
    rewrites: int = Field(ge=1, le=4)
    distance: int = Field(ge=8, le=24)
    example_seed: int = Field(ge=0)


@lru_cache(maxsize=64)
def _record(example_seed, distance, rewrites, layout):
    chosen, _ = cbl.sample_layouts(example_seed, 0, 4, distance, rewrites, [layout], 1)
    return chosen[layout][0]


def _which_write(prediction, values):
    for i, value in enumerate(values):
        if prediction == value:
            return 'latest' if i == len(values) - 1 else WRITE_NAMES[i]
    return 'other'


def _start_means(init, variant, seed):
    """Recorded start mean decay per layer for a mean-decay model (None for default)."""
    if init != 'mean-decay':
        return None
    path = shortcut.local_checkpoints()[(init, variant, seed)].parent / 'calibration.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    return [float(layer['achieved_mean_decay'][0][0]) if variant == 'scalar' else
            sum(layer['achieved_mean_decay'][0]) / len(layer['achieved_mean_decay'][0]) for layer in data['layers']]


@torch.no_grad()
def _run(model, record, req):
    cell = cbl.to_cell([record], req.rewrites, dict(example_seed=req.example_seed, distance=req.distance,
                                                   layout=req.layout))
    tokens = cell['tokens'][0]
    labels = cell['case'].data.labels[0]
    mask = labels != -100
    logits = model.scored_logits(tokens[None], mask[None])
    row = {pos: i for i, pos in enumerate(mask.nonzero().flatten().tolist())}
    probs = torch.softmax(logits, -1)
    preds = logits.argmax(-1)
    values = [int(v) for v in record.metadata['write_values']]

    queries = []
    for a in record.answers:
        i = row[a.label_position]
        gold = int(labels[a.label_position])
        pred = int(preds[i])
        source = _which_write(pred, values) if a.kind == 'latest' else ('latest' if pred == gold else 'other')
        top = torch.topk(probs[i], 5)
        queries.append(dict(position=int(a.label_position), kind=a.kind, gold=gold, prediction=pred,
                            which_write=source,
                            top5=[dict(token=int(t), prob=round(float(p), 4)) for p, t in zip(top.values, top.indices)]))

    # Decay values of every layer for this exact example, from the same mixer inputs the recurrence uses.
    captured, _ = internals.mixer_inputs(model, tokens[None])
    layers = []
    for idx, block in enumerate(model.network.backbone.layers):
        alpha = block.sequence_mixer.decay(captured[idx])[0, :, 0, :]      # [T, gate_width]
        layers.append(dict(layer=idx + 1, alpha=[[round(float(v), 4) for v in row_] for row_ in alpha.tolist()]))

    return dict(
        example=dict(tokens=[int(t) for t in tokens.tolist()], roles=shortcut.roles_for(record, tokens.tolist()),
                     pattern=cbl.pattern_of(record), write_values=values, queries=queries),
        alpha=dict(layers=layers),
        model=dict(init=req.init, variant=req.variant, seed=req.seed,
                   parameters=sum(p.numel() for p in model.parameters()),
                   checkpoint=repo_relative(shortcut.local_checkpoints()[(req.init, req.variant, req.seed)])),
        start_mean=_start_means(req.init, req.variant, req.seed),
    )


@router.post('/try')
def try_example(req: TryRequest):
    from app import _lock   # the server-wide lock: the shortcut page patches shared models while it holds it
    if req.layout not in shortcut.allowed_layouts(req.rewrites):
        raise HTTPException(422, f"layout '{req.layout}' is not valid for rewrites={req.rewrites}")
    path = shortcut.local_checkpoints().get((req.init, req.variant, req.seed))
    if path is None:
        raise HTTPException(404, f'no local overwrite checkpoint for {req.init}/{req.variant}/seed {req.seed}')
    with _lock:
        try:
            record = _record(req.example_seed, req.distance, req.rewrites, req.layout)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(422, f'could not sample this layout: {exc}')
        model = shortcut._load(str(path), req.variant, req.seed)
        return _run(model, record, req)
