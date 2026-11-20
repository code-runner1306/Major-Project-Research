"""Longer sequences and filler decay page: saved summaries, local MQAR models, and a live run.

GET  /api/length/summary   saved length sweep (36 cells), targeted replays, checkpoint audit
GET  /api/length/decay     saved filler decay diagnostic (per layer and granularity) and product histogram
GET  /api/length/models    the 12 local MQAR checkpoints
POST /api/length/run       live run on CPU: accuracy at a chosen filler count and the decay-only survival curve
"""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import length_data

router = APIRouter()


class RunRequest(BaseModel):
    init: Literal['default', 'mean-decay']
    variant: Literal['scalar', 'channel-wise']
    seed: Literal[42, 43, 44]
    filler: int = Field(ge=0, le=length_data.MAX_FILLER)
    examples: int = Field(default=8, ge=1, le=length_data.MAX_EXAMPLES)


def _fail(exc: length_data.LengthError):
    raise HTTPException(exc.status, exc.detail)


@router.get('/api/length/summary')
def length_summary():
    try:
        return length_data.summary()
    except length_data.LengthError as exc:
        _fail(exc)


@router.get('/api/length/decay')
def length_decay():
    try:
        return length_data.decay()
    except length_data.LengthError as exc:
        _fail(exc)


@router.get('/api/length/models')
def length_models():
    return length_data.models()


@router.post('/api/length/run')
def length_run(req: RunRequest):
    try:
        return length_data.run(req.init, req.variant, req.seed, req.filler, req.examples)
    except length_data.LengthError as exc:
        _fail(exc)
