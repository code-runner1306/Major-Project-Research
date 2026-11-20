"""Selective reset pilot page: saved results, the checkpoint's answer profile, and one live try-it."""
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import selective_reset_data as data

router = APIRouter()


class TryRequest(BaseModel):
    load: Literal[1, 2, 4, 8]
    reset: bool
    query: Literal['target', 'survivor', 'rewrite', 'absent']
    seed: int = Field(ge=0, le=99_999)


@router.get('/api/selective-reset/summary')
def selective_reset_summary():
    return data.summary()


@router.get('/api/selective-reset/profile')
def selective_reset_profile():
    if not data.is_live():
        raise HTTPException(409, 'no checkpoint for the pilot on this machine')
    return data.profile()


@router.post('/api/selective-reset/try')
def selective_reset_try(req: TryRequest):
    if not data.is_live():
        raise HTTPException(409, 'no checkpoint for the pilot on this machine')
    try:
        return data.try_example(req.load, req.reset, req.query, req.seed)
    except Exception as exc:  # a clear message instead of a silent failure
        raise HTTPException(500, f'could not run the checkpoint: {exc}')
