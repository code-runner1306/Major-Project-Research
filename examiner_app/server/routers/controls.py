"""Negative controls page: saved Zoology and KATA results, live Zoology try-it, and the saved slot-histogram export."""
import json
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import controls_data

router = APIRouter(prefix='/api/controls', tags=['controls'])
_lock = threading.Lock()   # one Zoology forward pass at a time; cached models are shared
INSIDE_FILE = Path(__file__).resolve().parents[1] / 'data' / 'controls_inside.json'


class ZoologyEvaluateRequest(BaseModel):
    init: Literal['default', 'mean-decay', 'published']
    variant: Literal['scalar', 'channel-wise']
    seed: Literal[42, 43, 44]
    layout: str
    examples: int = Field(ge=1, le=controls_data.TRY_MAX_EXAMPLES)
    example_seed: int = Field(ge=0)
    show_examples: int = Field(default=3, ge=0, le=8)


@router.get('/summary')
def summary():
    return controls_data.summary()


@router.get('/zoology/models')
def zoology_models():
    return controls_data.zoology_models()


@router.post('/zoology/evaluate')
def zoology_evaluate(req: ZoologyEvaluateRequest):
    with _lock:
        try:
            return controls_data.evaluate(req.init, req.variant, req.seed, req.layout, req.examples,
                                          req.example_seed, req.show_examples)
        except controls_data.ControlsError as exc:
            raise HTTPException(exc.status, exc.detail)


@router.get('/inside')
def inside():
    if not INSIDE_FILE.is_file():
        raise HTTPException(404, 'controls_inside.json has not been generated; run server/controls_export.py')
    return json.loads(INSIDE_FILE.read_text(encoding='utf-8'))
