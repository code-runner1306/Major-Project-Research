import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402

client = TestClient(app)

TRY_BODY = dict(arm='A0_control', variant='scalar', seed=42, layout='block_first', rewrites=3,
                distance=8, example_seed=0)


def try_example(**overrides):
    return client.post('/api/layout-fix/try', json={**TRY_BODY, **overrides})


def test_health_counts_thirty_checkpoints():
    r = client.get('/api/health')
    assert r.status_code == 200
    body = r.json()
    assert body['ok'] is True
    assert body['checkpoints'] == 30


def test_runs_lists_thirty_entries():
    r = client.get('/api/runs')
    assert r.status_code == 200
    runs = r.json()
    assert len(runs) == 30
    assert {'arm', 'variant', 'seed', 'path', 'final_dev_accuracy'} <= set(runs[0])


def test_summary_has_fresh_results_and_provenance():
    body = client.get('/api/layout-fix/summary').json()
    assert 'fresh_results' in body
    assert body['provenance']['generated'] == body['meta']['generated']


def test_try_block_first_scalar_shape_and_speed():
    start = time.perf_counter()
    r = try_example()
    elapsed = time.perf_counter() - start
    assert r.status_code == 200, r.text
    assert elapsed < 5, f'try took {elapsed:.2f}s'
    body = r.json()
    example, internals = body['example'], body['internals']
    assert len(example['queries']) == 2
    assert len(example['roles']) == len(example['tokens'])
    assert len(internals['layers']) == 4
    assert all(len(row) == 1 for row in internals['layers'][0]['alpha'])
    assert len(internals['state_crops']) == len(example['tokens'])
    assert internals['recurrence_check_max_abs_diff'] < 1e-5
    assert body['model']['arm'] == 'A0_control'


def test_try_channel_wise_alpha_width_128():
    r = try_example(variant='channel-wise')
    assert r.status_code == 200, r.text
    alpha = r.json()['internals']['layers'][0]['alpha']
    assert all(len(row) == 128 for row in alpha)


def test_layout_invalid_for_rewrites_is_422():
    r = try_example(layout='initial_then_run', rewrites=1)
    assert r.status_code == 422


@pytest.mark.parametrize('override', [dict(arm='A9_nope'), dict(variant='diagonal'), dict(seed=7),
                                      dict(layout='not_a_layout'), dict(rewrites=5), dict(distance=4)])
def test_invalid_inputs_are_422(override):
    assert try_example(**override).status_code == 422


def test_code_symbol_method_returns_source():
    r = client.get('/api/code', params=dict(path='recipe_model.py', symbol='RecipeGDN.decay'))
    assert r.status_code == 200, r.text
    body = r.json()
    assert 'def decay' in body['source']
    assert body['start_line'] <= body['end_line']


def test_code_parent_traversal_is_400():
    r = client.get('/api/code', params=dict(path='../x.py', symbol='f'))
    assert r.status_code == 400


def test_code_absolute_path_is_400():
    r = client.get('/api/code', params=dict(path='C:/Windows/win.ini', symbol='f'))
    assert r.status_code == 400


def test_code_missing_symbol_is_404():
    r = client.get('/api/code', params=dict(path='recipe_model.py', symbol='NoSuchThing'))
    assert r.status_code == 404


def test_code_lines_limit_is_enforced():
    r = client.get('/api/code/lines', params=dict(path='recipe_model.py', start=1, end=200))
    assert r.status_code == 400
    r = client.get('/api/code/lines', params=dict(path='recipe_model.py', start=1, end=10))
    assert r.status_code == 200
