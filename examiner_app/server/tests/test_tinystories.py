import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402

client = TestClient(app)


def test_summary_measures_only_10m_and_30m():
    body = client.get('/api/tinystories/summary').json()
    sizes = {row['size'] for row in body['measured']}
    assert sizes == {'10M', '30M'}, 'the 55M folder is a copy of 30M and must never be a measured size'
    thirty = {row['variant']: row for row in body['measured'] if row['size'] == '30M'}
    assert thirty['scalar']['seeds'] == [42, 43, 44]
    assert thirty['channel-wise']['seeds'] == [42, 43]
    assert thirty['scalar']['params'] == 29_272_320


def test_das_rows_are_starred_and_not_measured():
    body = client.get('/api/tinystories/summary').json()
    assert body['das'], 'Das rows should be parsed from decay_design_results.md'
    assert all(row['label'].endswith('*') for row in body['das'])
    assert all(row['size'] in {'10M', '55M', '100M'} for row in body['das'])
    assert not any(row['size'] == '55M' for row in body['measured'])


def test_paired_gaps_are_computed_from_metrics():
    body = client.get('/api/tinystories/summary').json()
    gaps = {(p['size'], p['seed']): p['channel_minus_scalar'] for p in body['paired']}
    assert ('30M', 42) in gaps and ('30M', 43) in gaps
    assert gaps[('30M', 43)] == pytest.approx(2.378652598261833 - 2.359882968664169, abs=1e-9)


def test_models_report_live_30m_checkpoints():
    rows = client.get('/api/tinystories/models').json()
    assert {r['variant'] for r in rows} == {'scalar', 'channel-wise'}
    assert all(r['size'] == '30M' and r['seed'] == 42 for r in rows)


def test_continue_returns_both_variants():
    res = client.post('/api/tinystories/continue', json={'prompt': 'Once upon a time, a little dog', 'max_new_tokens': 5})
    assert res.status_code == 200, res.text
    body = res.json()
    assert set(body['variants']) == {'scalar', 'channel-wise'}
    assert all(len(v['tokens']) <= 5 for v in body['variants'].values())


def test_gates_are_per_layer_and_hybrid_layers_listed():
    res = client.post('/api/tinystories/gates', json={'variant': 'channel-wise', 'text': 'The dog ran to the park.'})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body['attention_layers'] == [4, 8]
    assert len(body['layers']) == 8
    assert body['gate_width'] == 40
    n = len(body['tokens'])
    assert all(len(layer['mean']) == n for layer in body['layers'])
    assert all(0 <= v <= 1 for layer in body['layers'] for v in layer['mean'])


def test_continue_rejects_empty_prompt():
    res = client.post('/api/tinystories/continue', json={'prompt': '', 'max_new_tokens': 5})
    assert res.status_code == 422


def test_sample_is_saved_from_local_parquet():
    body = client.get('/api/tinystories/sample').json()
    assert body['split'] == 'validation'
    assert len(body['stories']) == 5
    assert all(s['tokens'] for s in body['stories'])
