"""FastAPI server for the examiner app. Run from examiner_app/server:

    ../../.venv/Scripts/python.exe -m uvicorn app:app --port 8765

Serves saved artifacts (layout-fix summary, run list, source code) and live CPU evaluation of the real
layout-fix checkpoints on fresh overwrite examples (/api/layout-fix/try). Nothing is trained or written.
"""
import json
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import confirm_block_layout as cbl  # noqa: E402
import checkpoints  # noqa: E402
import codeview  # noqa: E402
import dataset_generators as datasets  # noqa: E402
import internals  # noqa: E402
import shortcut  # noqa: E402
from checkpoints import load_model, all_runs, repo_relative  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(4)
# Fresh evaluation data: a seed range no training or earlier experiment used.
cbl.SEED_OFFSET = 3_000_000

SUMMARY_FILE = ROOT / 'examiner_app' / 'server' / 'data' / 'layout_fix_export.json'
WRITE_NAMES = ('initial', 'first_rewrite', 'second_rewrite', 'third_rewrite')
LAYOUTS = ('block_first', 'general_nonblock', 'block_after_one_unrelated', 'run_broken_before_latest',
           'initial_then_run')
ArmName = Literal['A0_control', 'A1_varied_distance', 'A2_no_block_first', 'A3_mixed_rewrites',
                  'A4_distance_and_rewrites']

app = FastAPI(title='Examiner API')
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5180'], allow_methods=['*'],
                   allow_headers=['*'])
_lock = threading.Lock()   # one forward pass at a time; the cached models are shared


def _include_page_routers():
    """Include every server/routers/*.py that exposes `router` (one module per experiment page)."""
    import importlib, pkgutil
    from routers import __path__ as router_path
    for info in pkgutil.iter_modules(router_path):
        module = importlib.import_module(f'routers.{info.name}')
        if hasattr(module, 'router'):
            app.include_router(module.router)


_include_page_routers()


class TryRequest(BaseModel):
    arm: ArmName
    variant: Literal['scalar', 'channel-wise']
    seed: Literal[42, 43, 44]
    layout: Literal['block_first', 'general_nonblock', 'block_after_one_unrelated',
                    'run_broken_before_latest', 'initial_then_run']
    rewrites: int = Field(ge=1, le=4)
    distance: int = Field(ge=8, le=24)
    example_seed: int = Field(ge=0)


def allowed_layouts(rewrites):
    return {'general_nonblock', *cbl.special_patterns(rewrites)}


@lru_cache(maxsize=64)
def sample_record(example_seed, distance, rewrites, layout):
    chosen, _ = cbl.sample_layouts(example_seed, 0, 4, distance, rewrites, [layout], 1)
    return chosen[layout][0]


def which_write(prediction, values):
    for i, value in enumerate(values):
        if prediction == value:
            return 'latest' if i == len(values) - 1 else WRITE_NAMES[i]
    return 'other'


def roles_for(record, tokens):
    """Same rule as describe_example in legacy_prototype/export_data.py."""
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


def run_example(model, record, req):
    # Answer-free input exactly as confirm_block_layout.to_cell builds it (gold after each query -> token 1).
    cell = cbl.to_cell([record], req.rewrites,
                       dict(example_seed=req.example_seed, distance=req.distance, layout=req.layout))
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
        if a.kind == 'latest':
            source = which_write(pred, values)
        else:
            source = 'latest' if pred == gold else 'other'
        top = torch.topk(probs[i], 5)
        queries.append(dict(
            position=int(a.label_position), kind=a.kind, gold=gold, prediction=pred, which_write=source,
            top5=[dict(token=int(t), prob=round(float(p), 4)) for p, t in zip(top.values, top.indices)]))

    toks = [int(t) for t in tokens.tolist()]
    trace = internals.trace(model, tokens)
    write_positions = [int(p) for p in record.metadata['write_positions']]
    return dict(
        example=dict(tokens=toks, roles=roles_for(record, toks), pattern=cbl.pattern_of(record),
                     write_values=values, queries=queries),
        internals=dict(layers=trace['layers'], state_crops=trace['state_crops'],
                       write_positions=write_positions, latest_write_position=write_positions[-1],
                       recurrence_check_max_abs_diff=trace['recurrence_check_max_abs_diff']),
        model=dict(arm=req.arm, variant=req.variant, seed=req.seed,
                   parameters=sum(p.numel() for p in model.parameters()),
                   checkpoint=repo_relative(checkpoints.find_checkpoint(req.arm, req.variant, req.seed))),
    )


@app.get('/api/health')
def health():
    return dict(ok=True, torch=torch.__version__, checkpoints=len(all_runs()))


@app.get('/api/layout-fix/summary')
def layout_fix_summary():
    data = json.loads(SUMMARY_FILE.read_text(encoding='utf-8'))
    data['provenance'] = dict(file=repo_relative(SUMMARY_FILE), generated=data['meta']['generated'])
    return data


@app.get('/api/runs')
def runs():
    out = []
    for arm, variant, seed, ckpt in all_runs():
        summary = json.loads((ckpt.parent / 'summary.json').read_text(encoding='utf-8'))
        out.append(dict(arm=arm, variant=variant, seed=seed, path=repo_relative(ckpt.parent),
                        final_dev_accuracy=summary['final_development']['answer_free']['exact_query_accuracy']))
    return out


@app.post('/api/layout-fix/try')
def layout_fix_try(req: TryRequest):
    if req.layout not in allowed_layouts(req.rewrites):
        raise HTTPException(422, f"layout '{req.layout}' is not valid for rewrites={req.rewrites}")
    with _lock:
        try:
            record = sample_record(req.example_seed, req.distance, req.rewrites, req.layout)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(422, f'could not sample this layout: {exc}')
        if checkpoints.find_checkpoint(req.arm, req.variant, req.seed) is None:
            raise HTTPException(404, f'no checkpoint for {req.arm}/{req.variant}/seed {req.seed}')
        with torch.no_grad():   # grad mode is thread-local; worker threads do not inherit set_grad_enabled
            return run_example(load_model(req.arm, req.variant, req.seed), record, req)


class ShortcutEvaluateRequest(BaseModel):
    init: Literal['default', 'mean-decay']
    variant: Literal['scalar', 'channel-wise']
    seed: Literal[42, 43, 44]
    layout: str
    rewrites: int = Field(ge=1, le=4)
    distance: int = Field(ge=8, le=24)
    examples: int = Field(ge=1, le=64)
    example_seed: int = Field(ge=0)
    patch: Literal['none', 'beta_one_latest', 'key_align_latest', 'beta_zero_rewrite1'] = 'none'
    show_examples: int = Field(default=shortcut.EXAMPLES_SHOWN, ge=1, le=shortcut.EXAMPLES_MAX)


@app.get('/api/shortcut/summary')
def shortcut_summary():
    return shortcut.summary()


@app.get('/api/shortcut/models')
def shortcut_models():
    return shortcut.models()


@app.post('/api/shortcut/evaluate')
def shortcut_evaluate(req: ShortcutEvaluateRequest):
    with _lock:   # cached models are shared; patched runs temporarily alter the GDN forward
        try:
            return shortcut.evaluate(req.init, req.variant, req.seed, req.layout, req.rewrites, req.distance,
                                     req.examples, req.example_seed, req.patch, req.show_examples)
        except shortcut.ShortcutError as exc:
            raise HTTPException(exc.status, exc.detail)


@app.get('/api/code')
def code(path: str = Query(...), symbol: str = Query(...), context: int = Query(0, ge=0, le=200)):
    try:
        return codeview.symbol_source(path, symbol, context)
    except codeview.CodeViewError as exc:
        raise HTTPException(exc.status, exc.detail)


@app.get('/api/code/lines')
def code_lines(path: str = Query(...), start: int = Query(...), end: int = Query(...)):
    try:
        return codeview.line_source(path, start, end)
    except codeview.CodeViewError as exc:
        raise HTTPException(exc.status, exc.detail)


class DatasetRequest(BaseModel):
    generator: str
    seed: int = Field(ge=0)
    params: dict = Field(default_factory=dict)


@app.on_event('startup')
def warm_datasets():
    datasets.start_warm_thread()   # imports the generator modules in the background; first requests stay fast


@app.get('/api/datasets')
def dataset_catalogue():
    return datasets.catalogue()


@app.post('/api/datasets/generate')
def dataset_generate(req: DatasetRequest):
    try:
        return datasets.generate(req.generator, req.seed, req.params)
    except datasets.ParamError as exc:
        raise HTTPException(422, str(exc))
    except datasets.Unavailable as exc:
        raise HTTPException(409, str(exc))
