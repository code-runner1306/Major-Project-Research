"""Decision fine-tuning appendix: saved feasibility reports only.

Reads the two feasibility reports written by decision_feasibility.py (repo-root result.json and result (1).json)
at request time. Nothing is trained or evaluated here, and no fine-tuned checkpoint was saved, so no live endpoint exists.
Every number on the page comes from these files.
"""
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[3]

router = APIRouter(prefix='/api/decision')

# (run id, report file). The ids are internal; the page names runs by length in plain words.
REPORTS = (
    ('short', ROOT / 'result.json'),
    ('longer', ROOT / 'result (1).json'),
)
BASE_CHECKPOINT_GLOB = 'output/tinystories/30M/*/*/final_checkpoint.pt'


def _load(path: Path) -> dict:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f'Saved report not found: {path.name}')
    return json.loads(path.read_text(encoding='utf-8'))


def _keys(value, found: set) -> set:
    """Every dict key anywhere in a loaded report (used to check what the report can and cannot answer)."""
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key).lower())
            _keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _keys(child, found)
    return found


def _has_key(keys: set, *needles: str) -> bool:
    return any(needle in key for key in keys for needle in needles)


def _variant_row(item: dict) -> dict:
    steps = item.get('step_seconds') or []
    return {
        'variant': item['variant'],
        'first_train_loss': item['first_train_loss'],
        'last_train_loss': item['last_train_loss'],
        'mean_validation_soft_loss': item['mean_validation_soft_loss'],
        'steady_step_median_seconds': item['steady_step_median_seconds'],
        'peak_allocated_gib': item['peak_allocated_gib'],
        'peak_reserved_gib': item['peak_reserved_gib'],
        'timed_steps': len(steps),
    }


@router.get('/summary')
def summary():
    """Saved feasibility runs: losses, step times and memory per decay granularity, plus what the reports lack."""
    runs = []
    first_report = None
    keys: set = set()
    for run_id, path in REPORTS:
        report = _load(path)
        _keys(report, keys)
        first_report = first_report or report
        runs.append({
            'id': run_id,
            'file': path.name,
            'steps': report['results'][0]['steps'],
            'batch_size': report['results'][0]['batch_size'],
            'max_length': report['results'][0]['max_length'],
            'train_examples': report['results'][0]['train_examples'],
            'validation_examples': report['results'][0]['validation_examples'],
            'state_truncations': report['state_truncations'],
            'variants': [_variant_row(item) for item in report['results']],
        })
        if any(item['steps'] != report['results'][0]['steps'] for item in report['results']):
            raise HTTPException(status_code=500, detail='Variants inside one report disagree on steps')

    labels = first_report['labels']
    if len(labels) != 5:
        raise HTTPException(status_code=500, detail='Expected five choice labels in the saved report')

    base = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(BASE_CHECKPOINT_GLOB))
    return {
        'question': {
            'workflow': first_report['workflow'],
            'question_id': first_report['question_id'],
            'labels': labels,
        },
        'dataset': {
            'split': first_report['dataset_split'],
            'fingerprint': first_report['dataset_fingerprint'],
            'available_train_cases': first_report['available_train_cases'],
            'available_validation_cases': first_report['available_validation_cases'],
        },
        'device': first_report['device'],
        'torch': first_report['torch'],
        'runs': runs,
        # What the reports can and cannot answer, computed from their keys.
        'pick_comparison_recorded': _has_key(keys, 'accuracy', 'prediction', 'pick', 'chosen', 'argmax'),
        'pre_training_validation_recorded': _has_key(keys, 'baseline', 'before', 'pre_train', 'initial_validation'),
        # decision_feasibility.py only writes the report JSON; it has no torch.save call.
        'fine_tuned_weights_saved': False,
        'base_checkpoints_local': base,
        'provenance': {
            'files': [p.name for _, p in REPORTS],
            'script': 'decision_feasibility.py',
            'note': 'Reports sit at the repo root; DECISION_FEASIBILITY.md names output/decision_feasibility/result.json.',
        },
    }
