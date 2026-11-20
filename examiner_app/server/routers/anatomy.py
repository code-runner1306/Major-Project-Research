"""Anatomy harness page API: saved results only (no checkpoints on this machine) and one live generator example."""
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query

import anatomy_data
import dataset_generators as datasets

router = APIRouter(prefix='/api/anatomy')


@router.get('/summary')
def summary():
    try:
        return anatomy_data.summary()
    except anatomy_data.AnatomyDataError as exc:
        raise HTTPException(404, str(exc))


@router.get('/runs')
def runs(cell: Optional[Literal['E1', 'E4', 'E5']] = None,
         source: Optional[Literal['ours', 'published']] = None):
    try:
        rows = anatomy_data.runs()
    except anatomy_data.AnatomyDataError as exc:
        raise HTTPException(404, str(exc))
    return [r for r in rows if (cell is None or r['cell'] == cell) and (source is None or r['source'] == source)]


@router.get('/example')
def example(experiment: Literal['E4', 'E5'] = Query(...), seed: int = Query(0, ge=0)):
    """The real Anatomy generator (dataset_generators, anatomy_e4 / anatomy_e5) for one seed. E5 = E4 + collision."""
    gid = 'anatomy_e4' if experiment == 'E4' else 'anatomy_e5'
    try:
        out = dict(datasets.generate(gid, seed, {}))   # copy: the generator caches its result
    except datasets.ParamError as exc:
        raise HTTPException(422, str(exc))
    out['experiment'] = experiment
    return out
