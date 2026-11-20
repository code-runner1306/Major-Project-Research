"""Attention control page: saved results, and one real example from the diagnostic's generator."""
from fastapi import APIRouter, HTTPException, Query

import attention_control_data as data

router = APIRouter()


@router.get('/api/attention-control/summary')
def attention_control_summary():
    return data.summary()


@router.get('/api/attention-control/example')
def attention_control_example(index: int = Query(0, ge=0, le=511), seed: int = Query(42, ge=0)):
    try:
        return data.example(index, seed)
    except Exception as exc:  # generator errors become a clear 500 message, never a silent empty example
        raise HTTPException(500, f'could not build the example: {exc}')
