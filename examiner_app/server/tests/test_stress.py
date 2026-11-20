import sys
from pathlib import Path

from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402
import stress_data  # noqa: E402

client = TestClient(app)

TRY = dict(init='default', variant='scalar', seed=44, intervention='none', rewrites=3, distance=8,
           examples=32, example_seed=0, show=0)


def try_with(**overrides):
    return client.post('/api/stress/try', json={**TRY, **overrides})


# ------------------------------------------------------------------ saved results (no model needed)

def test_summary_has_all_stress_conditions():
    r = client.get('/api/stress/summary')
    assert r.status_code == 200, r.text
    body = r.json()
    conditions = body['stress']['conditions']
    assert len(conditions) == 32                      # 2 tasks x 2 inits x 2 granularities x 4 cells
    assert len(body['stress']['cells']) == 8
    assert len(body['stress']['per_seed']) == 96      # 24 models x 4 cells
    overwrite_latest = {(c['initialization'], c['variant']): c['latest']
                        for c in conditions if c['family'] == 'overwrite' and c['cell'] == 'more_rewrites'}
    # REPORT.md: newest-value recall with three rewrites, mean over seeds, 74.61 / 77.60 / 52.47 / 56.64 %
    assert overwrite_latest[('default', 'scalar')] == 0.7461
    assert overwrite_latest[('default', 'channel-wise')] == 0.776
    assert overwrite_latest[('mean-decay', 'scalar')] == 0.5247
    assert overwrite_latest[('mean-decay', 'channel-wise')] == 0.5664


def test_mqar_eight_keys_seed_44_scalar_default_matches_models_csv():
    body = client.get('/api/stress/summary').json()
    rows = [r for r in body['stress']['per_seed'] if r['family'] == 'mqar' and r['initialization'] == 'default'
            and r['variant'] == 'scalar' and r['seed'] == 44 and r['cell'] == 'more_keys']
    assert len(rows) == 1
    assert round(rows[0]['accuracy'] * 2048) == 1091          # REPORT.md: 1091/2048 correct


def test_write_patch_aggregates_match_tables_md():
    body = client.get('/api/stress/summary').json()['write_patches']
    patches = {p['id']: p for p in body['patches']}
    assert all(p['models'] == 18 for p in patches.values())
    assert all(p['local_models'] == 12 for p in patches.values())
    # tables.md: change vs unpatched at three rewrites, in points
    assert round(patches['beta_one_latest']['latest_change'] * 100, 2) == -0.80
    assert (patches['beta_one_latest']['improved'], patches['beta_one_latest']['worse']) == (0, 2)
    assert round(patches['beta_zero_rewrite1']['latest_change'] * 100, 2) == 4.12
    assert round(patches['key_align_latest']['latest_change'] * 100, 2) == 16.21
    assert round(patches['key_align_latest']['retention_change'] * 100, 2) == -18.40
    assert (patches['key_align_latest']['improved'], patches['key_align_latest']['worse']) == (14, 0)
    assert round(body['baseline']['latest'] * 100, 2) == 64.50


def test_shared_mean_ablation_matches_stronger_test_per_model():
    body = client.get('/api/stress/summary').json()['gate_interventions']
    ablation = {(r['family'], r['seed'], r['initialization']): r['shared_mean'] for r in body['shared_mean_per_model']}
    stronger = {(r['family'], r['seed'], r['initialization']): r['shared_mean'] for r in body['per_model']}
    assert len(ablation) == 12
    assert ablation.keys() == stronger.keys()
    for key in ablation:
        assert abs(ablation[key] - stronger[key]) < 1e-9
    means = {(r['family'], r['initialization']): r['shared_mean'] for r in body['condition_means']}
    assert round(means[('mqar', 'default')] * 100, 2) == 82.16      # stronger REPORT.md table
    assert round(means[('mqar', 'mean-decay')] * 100, 2) == 69.63


def test_models_lists_twelve_local_overwrite_checkpoints():
    r = client.get('/api/stress/models')
    assert r.status_code == 200
    assert len(r.json()) == 12


def test_interventions_listed():
    ids = [i['id'] for i in client.get('/api/stress/interventions').json()]
    assert ids == ['none', 'beta_one_latest', 'beta_zero_rewrite1', 'key_align_latest', 'alpha_shared_mean']


# ------------------------------------------------------------------ live try

def test_try_rejects_bad_settings():
    assert client.post('/api/stress/try', json={**TRY, 'rewrites': 5}).status_code == 422
    assert client.post('/api/stress/try', json={**TRY, 'distance': 40}).status_code == 422
    assert client.post('/api/stress/try', json={**TRY, 'intervention': 'delete_all'}).status_code == 422
    assert client.post('/api/stress/try', json={**TRY, 'show': 32}).status_code == 422


def test_try_none_leaves_predictions_unchanged():
    r = try_with()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['before'] == body['after']
    assert body['changed_answers'] == 0 and body['fixed'] == 0 and body['broken'] == 0
    assert body['model']['parameters'] > 1_000_000


def test_try_is_deterministic_and_restores_model():
    first = try_with().json()
    stress_data.try_run('default', 'channel-wise', 44, 'alpha_shared_mean', 3, 8, 16, 0, 0)  # patch, then undo
    second = try_with().json()
    assert first['before'] == second['before']
    assert first['example']['tokens'] == second['example']['tokens']


def test_alpha_shared_mean_has_no_effect_on_scalar_models():
    body = try_with(intervention='alpha_shared_mean').json()
    assert body['setup']['no_effect_expected'] is True
    assert body['changed_answers'] == 0
    for layer in body['gates']['after']:
        assert layer['alpha_min'] == layer['alpha_max']           # one decay value per head, nothing to average


def test_alpha_shared_mean_flattens_channel_gates():
    body = try_with(variant='channel-wise', intervention='alpha_shared_mean').json()
    assert body['setup']['no_effect_expected'] is False
    before, after = body['gates']['before'], body['gates']['after']
    assert any(b['alpha_min'] != b['alpha_max'] for layer in before for b in [layer])
    for layer in after:
        assert all(lo == hi for lo, hi in zip(layer['alpha_min'], layer['alpha_max']))


def test_example_shape_and_gate_lengths():
    body = try_with(rewrites=2, distance=12, show=3).json()
    tokens = body['example']['tokens']
    assert len(tokens) == len(body['example']['roles']) and len(tokens) > 40
    assert len(body['gates']['before']) == 4
    for layer in body['gates']['before']:
        assert len(layer['beta']) == len(layer['state_norm']) == len(tokens)
        assert all(x > 0 for x in layer['state_norm'][:10])
    assert body['example']['values'][-1] in [q['gold'] for q in body['example']['queries'] if q['kind'] == 'latest']
    assert body['setup']['show'] == 3
