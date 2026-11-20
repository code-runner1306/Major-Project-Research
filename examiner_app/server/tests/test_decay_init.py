import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402

client = TestClient(app)

TRY = dict(init='default', variant='scalar', seed=42, layout='general_nonblock', rewrites=1, distance=8,
           example_seed=7)


def _by(rows, init, variant):
    return next(r for r in rows if r['init'] == init and r['variant'] == variant)


def test_summary_overwrite_learning_means_match_report():
    r = client.get('/api/decay-init/summary')
    assert r.status_code == 200, r.text
    rows = r.json()['families']['overwrite']['measures']['learning']
    # three_seed_exploration_20261117/REPORT.md, overwrite table (scores on a 0 to 100 scale there).
    expected = {('default', 'scalar'): 91.00, ('default', 'channel-wise'): 83.80,
                ('mean-decay', 'scalar'): 82.23, ('mean-decay', 'channel-wise'): 82.51}
    for (init, variant), value in expected.items():
        assert round(_by(rows, init, variant)['mean'] * 100, 2) == pytest.approx(value, abs=0.011)


def test_summary_overwrite_interaction_per_seed_matches_report():
    r = client.get('/api/decay-init/summary')
    contrast = r.json()['families']['overwrite']['contrasts']['learning']
    # REPORT: interaction per seed -13.18, -1.59, +37.23 points (mean-decay gap minus default gap).
    seeds = [round(v * 100, 2) for v in contrast['interaction']['seeds']]
    assert seeds == pytest.approx([-13.18, -1.59, 37.23], abs=0.01)


def test_summary_fresh_accuracy_matches_confirmation_report():
    r = client.get('/api/decay-init/summary')
    rows = r.json()['families']['overwrite']['measures']['fresh']
    # three_seed_confirmation_20261117/REPORT.md, mean final primary accuracy (overwrite column).
    expected = {('default', 'scalar'): 99.3164, ('default', 'channel-wise'): 98.7956,
                ('mean-decay', 'scalar'): 97.9818, ('mean-decay', 'channel-wise'): 97.3633}
    for (init, variant), value in expected.items():
        assert _by(rows, init, variant)['mean'] * 100 == pytest.approx(value, abs=0.001)


def test_summary_lists_three_seeds_and_both_families():
    body = client.get('/api/decay-init/summary').json()
    assert body['seeds'] == [42, 43, 44]
    assert set(body['families']) == {'overwrite', 'mqar'}
    for family in body['families'].values():
        for rows in family['measures'].values():
            assert len(rows) == 4 and all(len(r['seeds']) == 3 for r in rows)


def test_models_are_the_twelve_local_overwrite_checkpoints():
    r = client.get('/api/decay-init/models')
    assert r.status_code == 200
    models = r.json()
    assert len(models) == 12
    assert {m['init'] for m in models} == {'default', 'mean-decay'}


def test_gates_have_four_layers_and_start_only_for_mean_decay():
    r = client.get('/api/decay-init/gates')
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body['models']) == 12
    for m in body['models']:
        assert [layer['layer'] for layer in m['layers']] == [1, 2, 3, 4]
        width = 1 if m['variant'] == 'scalar' else 128
        for layer in m['layers']:
            assert len(layer['trained']['per_channel']) == width
            assert all(0 < v < 1 for v in layer['trained']['per_channel'])
            if m['init'] == 'default':
                assert layer['start'] is None
            else:
                assert len(layer['start']['per_channel']) == width
                assert layer['start']['maximum_error'] <= 1e-6


def test_mean_decay_start_matches_half_life_targets():
    body = client.get('/api/decay-init/gates').json()
    for m in body['models']:
        if m['init'] != 'mean-decay':
            continue
        for layer in m['layers']:
            target = 2 ** (-1 / layer['start']['half_life'])
            assert layer['start']['mean'] == pytest.approx(target, abs=1e-6)


def test_try_returns_example_prediction_and_decay_per_token():
    r = client.post('/api/decay-init/try', json=TRY)
    assert r.status_code == 200, r.text
    body = r.json()
    tokens = body['example']['tokens']
    assert len(body['example']['roles']) == len(tokens)
    assert len(body['alpha']['layers']) == 4
    for layer in body['alpha']['layers']:
        assert len(layer['alpha']) == len(tokens)
        assert all(len(row) == 1 for row in layer['alpha'])
        assert all(0 < row[0] < 1 for row in layer['alpha'])
    kinds = {q['kind'] for q in body['example']['queries']}
    assert 'latest' in kinds
    assert body['model']['parameters'] == 1585416
    assert body['start_mean'] is None


def test_try_channel_wise_mean_decay_has_start_means_per_layer():
    r = client.post('/api/decay-init/try', json={**TRY, 'init': 'mean-decay', 'variant': 'channel-wise', 'seed': 43})
    assert r.status_code == 200, r.text
    body = r.json()
    assert all(len(row) == 128 for row in body['alpha']['layers'][0]['alpha'])
    assert len(body['start_mean']) == 4
    assert body['model']['parameters'] == 1651456


def test_try_rejects_layout_not_valid_for_rewrites():
    # The initial-then-run layout exists only with two or more rewrites.
    r = client.post('/api/decay-init/try', json={**TRY, 'layout': 'initial_then_run', 'rewrites': 1})
    assert r.status_code == 422


def test_try_is_deterministic_for_one_example_seed():
    a = client.post('/api/decay-init/try', json=TRY).json()
    b = client.post('/api/decay-init/try', json=TRY).json()
    assert a['example']['queries'] == b['example']['queries']
    assert a['alpha'] == b['alpha']
