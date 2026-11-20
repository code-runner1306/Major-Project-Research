"""Saved results of the Anatomy harness (third_party/dualgoose), read at request time. Nothing is trained or loaded.

Our runs: results/mqar_probe and results/long_retrieval (armed_gdn = channel-wise decay, armed_gdn_scalar = scalar).
Published runs: upstream_results/ (the authors' numbers, unchanged). Mechanism runs: results/decay_interventions.
Verdicts and per-cell tests are copied from results/granularity_report.json, which scripts/analyze_granularity.py wrote.
"""
import json
import statistics
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DG = ROOT / 'third_party' / 'dualgoose'
PREDICTIONS = DG / 'PREDICTIONS_granularity.json'
REPORT = DG / 'results' / 'granularity_report.json'
OURS_DIR = DG / 'results'
UPSTREAM_DIR = DG / 'upstream_results'


class AnatomyDataError(Exception):
    pass


# Our cells: folder and file-name suffix written by run_A6000_dualgoose_granularity.ps1 (E1 = K=32 MQAR probe).
CELLS = {
    'E1': dict(title='Masked recall, 32 pairs', folder='mqar_probe', ours_suffix='k32_d32l1_gran',
               upstream_tag='k32_d32l1_h10', threshold_key='mqar_lock_in', collision=False),
    'E4': dict(title='Recall across a distractor haystack', folder='long_retrieval', ours_suffix='L64p4_d32_granE4',
               upstream_tag='L64p4_d32_h5_d4c', threshold_key='long_retrieval_lock_in', collision=False),
    'E5': dict(title='Recall with colliding keys', folder='long_retrieval', ours_suffix='L64p4_d32_granE5',
               upstream_tag='L64p4_d32_h10ck', threshold_key='long_retrieval_lock_in', collision=True),
}

# Published arms per cell, using the same config tag as the upstream reference for that cell.
UPSTREAM_ARMS = {
    'E1': ['armed_gdn', 'gated_deltanet', 'deltanet', 'mamba2_ref'],
    'E4': ['armed_gdn', 'armed_gdn_fwd', 'attention'],
    'E5': ['armed_gdn', 'armed_gdn_fwd', 'gated_deltanet'],
}
UPSTREAM_LABEL = {
    'armed_gdn': 'Published channel-wise decay, with convolution',
    'gated_deltanet': 'Published channel-wise decay, no convolution',
    'armed_gdn_fwd': 'Published, forward stream only',
    'deltanet': 'Published, no decay, no convolution',
    'mamba2_ref': 'Published Mamba-2 reference (diagonal, with convolution)',
    'attention': 'Published attention (reference)',
}
UPSTREAM_DECAY = {'armed_gdn': 'channel-wise', 'gated_deltanet': 'channel-wise', 'armed_gdn_fwd': 'channel-wise',
                  'deltanet': 'none', 'mamba2_ref': 'none', 'attention': 'none'}
OUR_LABEL = {
    'armed_gdn': 'Channel-wise decay (our reproduction)',
    'armed_gdn_scalar': 'Scalar decay (ours, added)',
}
OUR_DECAY = {'armed_gdn': 'channel-wise', 'armed_gdn_scalar': 'scalar'}
INTERVENTIONS = {
    'original': 'Trained channel-wise decay, unchanged',
    'fixed_channel_mean': "Each channel's decay fixed to its average",
    'token_channel_average': "Each token's channel decays replaced by their average",
    'fixed_scalar_mean': 'One constant decay for every channel and token',
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _threshold(cell_id: str) -> float:
    return float(_load(PREDICTIONS)['thresholds'][CELLS[cell_id]['threshold_key']])


def _run_row(cell_id, source, arm, label, decay, path, d, threshold) -> dict:
    acc = float(d['eval_accuracy'])
    rebind = d.get('rebind_accuracy')
    git = d.get('git')
    return dict(
        cell=cell_id, source=source, arm=arm, label=label, decay=decay, seed=int(d['seed']),
        eval_accuracy=acc, rebind_accuracy=None if rebind is None else float(rebind),
        lock_in=acc >= threshold, value_floor=d.get('value_marginal_floor'),
        commit=git.get('commit') if isinstance(git, dict) else None,
        file=_rel(path),
    )


def _matching(folder: Path, stem: str) -> list[tuple[Path, dict]]:
    """Files `<stem>_seed<N>.json` in a folder, sorted by seed. One file is one run."""
    items = [(p, _load(p)) for p in folder.glob(f'{stem}_seed*.json')]
    return sorted(items, key=lambda item: item[1]['seed'])


@lru_cache(maxsize=1)
def runs() -> list[dict]:
    """Every saved run in the three cells: ours (channel-wise and scalar) and the published reference arms."""
    out = []
    for cell_id, cell in CELLS.items():
        threshold = _threshold(cell_id)
        for arm in ('armed_gdn', 'armed_gdn_scalar'):
            for path, d in _matching(OURS_DIR / cell['folder'], f'{arm}_{cell["ours_suffix"]}'):
                out.append(_run_row(cell_id, 'ours', arm, OUR_LABEL[arm], OUR_DECAY[arm], path, d, threshold))
        for arm in UPSTREAM_ARMS[cell_id]:
            for path, d in _matching(UPSTREAM_DIR / cell['folder'], f'{arm}_{cell["upstream_tag"]}'):
                out.append(_run_row(cell_id, 'published', arm, UPSTREAM_LABEL[arm], UPSTREAM_DECAY[arm], path, d,
                                    threshold))
    if not out:
        raise AnatomyDataError('no saved Anatomy runs found under third_party/dualgoose')
    return out


@lru_cache(maxsize=1)
def interventions() -> dict:
    """Decay interventions (E6) on the trained channel-wise checkpoints, per cell: mean over seeds and per seed."""
    by_cell: dict[str, list[dict]] = {}
    for path in sorted((OURS_DIR / 'decay_interventions').glob('*.json')):
        name = path.name
        cell = 'E1' if '_gran_seed' in name else 'E4' if '_granE4_' in name else 'E5' if '_granE5_' in name else None
        if cell is None:
            continue
        d = _load(path)
        by_cell.setdefault(cell, []).append(dict(seed=int(d['seed']), accuracy=d['accuracy'], file=_rel(path)))
    out = {}
    for cell in CELLS:
        items = sorted(by_cell.get(cell, []), key=lambda i: i['seed'])
        if not items:
            continue
        means = {k: statistics.fmean(i['accuracy'][k] for i in items) for k in INTERVENTIONS}
        out[cell] = dict(n=len(items), means=means, runs=[dict(seed=i['seed'], accuracy=i['accuracy'], file=i['file'])
                                                         for i in items])
    return out


@lru_cache(maxsize=1)
def saved_channel_decay() -> dict:
    """Saved per-channel decay averages of one trained channel-wise model per cell (seed 0, first calibration set)."""
    out = {}
    for cell in CELLS:
        for path in sorted((OURS_DIR / 'decay_interventions').glob('*.json')):
            name = path.name
            hit = ('_gran_seed0' in name and cell == 'E1') or (f'_gran{cell}_seed0' in name and cell != 'E1')
            if not hit:
                continue
            d = _load(path)
            means = d.get('decay_channel_means', {})
            first = next(iter(means.values()), None)
            if first is not None:
                out[cell] = dict(values=first, file=_rel(path))
    return out


def _arm_block(arm_id, label, decay, source, rows, threshold) -> dict:
    accs = [r['eval_accuracy'] for r in rows]
    rebinds = [r['rebind_accuracy'] for r in rows if r['rebind_accuracy'] is not None]
    return dict(
        id=arm_id, label=label, decay=decay, source=source, n=len(accs),
        mean=statistics.fmean(accs) if accs else None,
        std=statistics.pstdev(accs) if accs else None,
        lock_in=sum(a >= threshold for a in accs),
        per_seed=[dict(seed=r['seed'], accuracy=r['eval_accuracy']) for r in rows],
        rebind_mean=statistics.fmean(rebinds) if rebinds else None,
    )


def summary() -> dict:
    """Page summary: per cell the arms with per-seed accuracy, the saved tests, and the mechanism runs."""
    try:
        report = _load(REPORT)
    except FileNotFoundError as exc:
        raise AnatomyDataError(f'missing saved report {_rel(REPORT)}') from exc
    predictions = _load(PREDICTIONS)
    all_runs = runs()

    cells = []
    for cell_id, cell in CELLS.items():
        threshold = _threshold(cell_id)
        arms = []
        for arm in ('armed_gdn', 'armed_gdn_scalar'):
            rows = [r for r in all_runs if r['cell'] == cell_id and r['arm'] == arm and r['source'] == 'ours']
            arms.append(_arm_block(f'ours_{OUR_DECAY[arm]}', OUR_LABEL[arm], OUR_DECAY[arm], 'ours', rows, threshold))
        for arm in UPSTREAM_ARMS[cell_id]:
            rows = [r for r in all_runs if r['cell'] == cell_id and r['arm'] == arm and r['source'] == 'published']
            arms.append(_arm_block(f'published_{arm}', UPSTREAM_LABEL[arm], UPSTREAM_DECAY[arm], 'published', rows,
                                   threshold))
        floors = [r['value_floor'] for r in all_runs if r['cell'] == cell_id and r['value_floor'] is not None]
        cells.append(dict(id=cell_id, title=cell['title'], collision=cell['collision'], threshold=threshold,
                          value_floor=floors[0] if floors else None, arms=arms))

    tests = {}
    for cell_id, r in report['cells'].items():
        tests[cell_id] = dict(
            channel_mean=r['channel']['mean'], scalar_mean=r['scalar']['mean'],
            mean_difference=r['mean_difference'], permutation_p=r['permutation_p'],
            channel_lock_in=r['channel']['lock_in'], scalar_lock_in=r['scalar']['lock_in'],
            lock_in_fisher_p=r['lock_in_fisher_p'], n=r['channel']['n'],
        )
    commits = sorted({r['commit'] for r in all_runs if r['source'] == 'ours' and r['commit']})
    return dict(
        provenance=dict(
            upstream='JIBSIL/dualgoose commit 8ca624a (Boesch & Wee, arXiv 2609.16183)',
            files=[_rel(PREDICTIONS), _rel(REPORT), _rel(DG / 'UPSTREAM.md')],
            our_commits=commits,
            frozen=predictions['frozen'],
        ),
        predictions=predictions['predictions'],
        cells=cells,
        tests=tests,
        channel_ahead_cells=[c for c, t in tests.items() if t['channel_lock_in'] > t['scalar_lock_in']],
        interventions=interventions(),
        intervention_labels=INTERVENTIONS,
        channel_decay=saved_channel_decay(),
        checkpoints_available=(DG / 'checkpoints').exists(),
    )
