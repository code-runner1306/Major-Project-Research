"""TinyStories API: saved results, live 30M continuation and gate readout, and a saved story sample.

GET  /api/tinystories/summary   measured runs (10M, 30M only), Das et al. rows (labelled *), paired seed gaps
GET  /api/tinystories/models    which live 30M checkpoints are on disk
POST /api/tinystories/continue  greedy continuation of a story opening, both decay designs, CPU
POST /api/tinystories/gates     real decay gate (alpha) values per Gated DeltaNet layer for a text
GET  /api/tinystories/sample    the saved HuggingFace-style TinyStories sample (server/data/tinystories/sample.json)
"""
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import tinystories_data as tsd
import dataset_generators as datasets

router = APIRouter(prefix='/api/tinystories')
SAMPLE_FILE = Path(__file__).resolve().parents[1] / 'data' / 'tinystories' / 'sample.json'


class ContinueRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=tsd.TEXT_CHARS_MAX)
    max_new_tokens: int = Field(default=30, ge=1, le=tsd.NEW_TOKENS_MAX)


class GatesRequest(BaseModel):
    variant: Literal['scalar', 'channel-wise']
    text: str = Field(min_length=1, max_length=tsd.TEXT_CHARS_MAX)


def _unavailable(exc):
    return HTTPException(status_code=409, detail=str(exc))


@router.get('/summary')
def summary():
    return tsd.summary()


@router.get('/models')
def models():
    return tsd.live_models()


@router.post('/continue')
def continue_story(body: ContinueRequest):
    try:
        return tsd.continue_text(body.prompt, body.max_new_tokens)
    except datasets.Unavailable as exc:
        raise _unavailable(exc)
    except FileNotFoundError as exc:
        raise _unavailable(exc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post('/gates')
def gates(body: GatesRequest):
    try:
        return tsd.gate_readout(body.variant, body.text)
    except datasets.Unavailable as exc:
        raise _unavailable(exc)
    except FileNotFoundError as exc:
        raise _unavailable(exc)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get('/sample')
def sample():
    if not SAMPLE_FILE.is_file():
        raise HTTPException(status_code=409, detail=f'sample file is missing: {SAMPLE_FILE.name}; run make_tinystories_sample.py')
    return json.loads(SAMPLE_FILE.read_text(encoding='utf-8'))
