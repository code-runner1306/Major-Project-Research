import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402
import controls_data  # noqa: E402

client = TestClient(app)


def summary():
    r = client.get('/api/controls/summary')
    assert r.status_code == 200, r.text
    return r.json()


def test_zoology_models_fourteen_all_local_two_published():
    models = client.get('/api/controls/zoology/models').json()
    assert len(models) == 14
    assert all(m['local'] for m in models)
    assert sum(m['group'] == 'published' for m in models) == 2
    assert {m['init'] for m in models} == {'default', 'mean-decay', 'published'}


def test_zoology_shortcut_count_matches_saved_verdict():
    z = summary()['zoology']
    saved = z['verdicts']['Z2_position_shortcut']
    assert z['shortcut_count'] == saved['models_showing'] == 1
    assert z['models_total'] == saved['of'] == 14
    flagged = [m for m in z['models'] if m['shortcut']]
    assert [f"{m['init']}/{m['variant']}/{m['seed']}" for m in flagged] == ['default/scalar/44']


def test_zoology_standard_layout_is_near_perfect_for_every_model():
    z = summary()['zoology']
    assert min(m['accuracy']['shift_0'] for m in z['models']) >= 0.99


def test_kata_is_results_only_with_three_runs_and_no_shortcut():
    k = summary()['kata']
    assert k['local_checkpoints'] == 0
    assert k['runs_total'] == 3
    assert k['shortcut_runs'] == 0
    assert k['parameters']['ours'] == 1167876


def test_saved_cell_is_reproduced_by_live_evaluation():
    # Seed-44 model on one blank token, 512 examples, example seed 44: the saved test's own data for seed 44.
    saved = next(m for m in summary()['zoology']['models'] if (m['init'], m['variant'], m['seed']) == ('default', 'scalar', 44))
    live = controls_data.evaluate('default', 'scalar', 44, 'shift_1', 512, 44, show_examples=0)
    assert abs(live['accuracy'] - saved['accuracy']['shift_1']) < 1e-9
    assert live['n_answers'] == 512 * 4


def test_try_endpoint_shape_and_counts():
    body = dict(init='default', variant='scalar', seed=44, layout='shift_1', examples=16, example_seed=44, show_examples=2)
    r = client.post('/api/controls/zoology/evaluate', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['n_answers'] == 16 * 4
    assert len(out['accuracy_by_query']) == 4
    assert len(out['shown']) == 2
    assert all(len(e['queries']) == 4 for e in out['shown'])
    assert abs(sum(out['answer_sources'].values()) - 1) < 1e-9
    assert out['accuracy'] == out['answer_sources']['correct']
    assert out['errors']['total'] == out['n_answers'] - round(out['accuracy'] * out['n_answers'])


def test_try_rejects_unknown_layout_and_bad_counts():
    base = dict(init='default', variant='scalar', seed=44, layout='shift_1', examples=4, example_seed=1, show_examples=1)
    assert client.post('/api/controls/zoology/evaluate', json={**base, 'layout': 'nope'}).status_code == 422
    assert client.post('/api/controls/zoology/evaluate', json={**base, 'examples': 65}).status_code == 422


def test_inside_export_matches_saved_results():
    r = client.get('/api/controls/inside')
    assert r.status_code == 200, r.text
    data = r.json()
    assert data['all_zoology_match_saved'] is True
    assert len(data['zoology']) == 14 * 2
    assert len(data['shortcut']) == 12
    assert all(len(row['pick_histogram']) >= 1 for row in data['shortcut'])
    json.dumps(data)
