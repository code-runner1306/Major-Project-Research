"""Cost and runtime page: saved timing (A6000, Kaggle T4) and live parameter counts of the recipe models."""
import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

import cost_data

router = APIRouter()

EXPORT = Path(__file__).resolve().parents[1] / 'data' / 'cost_export.json'


@lru_cache(maxsize=1)
def _export():
    if not EXPORT.exists():
        raise HTTPException(500, 'cost_export.json is missing; run server/build_cost_export.py')
    return json.loads(EXPORT.read_text(encoding='utf-8'))


@router.get('/api/cost/summary')
def cost_summary():
    data = dict(_export())
    data['provenance'] = {'file': 'examiner_app/server/data/cost_export.json',
                          'generator': data['meta']['generator']}
    return data


@router.get('/api/cost/parameters')
def cost_parameters():
    try:
        return cost_data.parameters()
    except Exception as exc:  # a model that cannot be built becomes a clear message, not a silent empty card
        raise HTTPException(500, f'could not build the recipe models on CPU: {exc}')
