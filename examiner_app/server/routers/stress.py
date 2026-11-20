"""Stress tests and gate interventions page API. Mounted by app.py through routers/__init__ discovery.

GET  /api/stress/summary        saved stress-test, gate-intervention and write-patch results
GET  /api/stress/models         the 12 local overwrite checkpoints
GET  /api/stress/interventions  the interventions the try endpoint accepts, in plain English
POST /api/stress/try            fresh examples, unpatched and patched, on one local checkpoint
"""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import stress_data

router = APIRouter(prefix='/api/stress')


class TryRequest(BaseModel):
    init: Literal['default', 'mean-decay']
    variant: Literal['scalar', 'channel-wise']
    seed: Literal[42, 43, 44]
    intervention: Literal['none', 'beta_one_latest', 'beta_zero_rewrite1', 'key_align_latest', 'alpha_shared_mean']
    rewrites: int = Field(ge=1, le=4)
    distance: int = Field(ge=8, le=24)
    examples: int = Field(default=32, ge=1, le=64)
    example_seed: int = Field(ge=0)
    show: int = Field(default=0, ge=0, le=63)


@router.get('/summary')
def summary():
    return stress_data.summary()


@router.get('/models')
def models():
    return stress_data.local_models()


@router.get('/interventions')
def interventions():
    return stress_data.INTERVENTIONS


@router.post('/try')
def try_stress(req: TryRequest):
    if req.show >= req.examples:
        raise HTTPException(422, f'show ({req.show}) must be smaller than examples ({req.examples})')
    try:
        return stress_data.try_run(req.init, req.variant, req.seed, req.intervention, req.rewrites, req.distance,
                                   req.examples, req.example_seed, req.show)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
