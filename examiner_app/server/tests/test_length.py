import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402

client = TestClient(app)
ROOT = SERVER.parents[1]
SWEEP = ROOT / 'research_results' / 'mqar_length_extension_20261119' / 'results.json'
DECAY = ROOT / 'research_results' / 'filler_decay_diagnostic_20261119' / 'results.json'


def test_models_lists_twelve_local_mqar_checkpoints():
    r = client.get('/api/length/models')
    assert r.status_code == 200, r.text
    models = r.json()
    assert len(models) == 12
    assert {m['init'] for m in models} == {'default', 'mean-decay'}
    assert {m['variant'] for m in models} == {'scalar', 'channel-wise'}
    assert {m['seed'] for m in models} == {42, 43, 44}
    assert all(m['checkpoint'].endswith('latest_checkpoint.pt') and '/mqar/' in m['checkpoint'] for m in models)


def test_summary_matches_saved_sweep_and_cross_checks():
    r = client.get('/api/length/summary')
    assert r.status_code == 200, r.text
    body = r.json()
    saved = json.loads(SWEEP.read_text(encoding='utf-8'))['results']
    assert body['lengths'] == [64, 256, 1024]
    assert len(body['cells']) == 36 == len(saved)
    by_key = {(c['init'], c['variant'], c['seed'], c['length']): c for c in body['cells']}
    for row in saved:
        init = 'mean-decay' if row['initialization'] == 'calibrated' else 'default'
        cell = by_key[(init, row['variant'], row['seed'], row['length'])]
        assert cell['correct'] == row['correct']
        assert cell['total'] == row['total'] == 1024
        assert abs(cell['accuracy'] - row['exact_query_accuracy']) < 1e-12
        assert cell['local'] is True
    assert body['verification']['verified_cells_agree'] == 36
    assert body['verification']['checkpoint_hashes'] == 12
    assert body['verification']['checkpoint_audit_status'] == 'complete'
    assert len(body['models']) == 12
    assert all(set(m['accuracy']) == {'64', '256', '1024'} for m in body['models'])
    means = {m['length']: m['mean'] for m in body['means']}
    assert means[64] > means[256] > means[1024]
    assert len(body['replays']) == 2
    assert all(rp['matches_sweep'] for rp in body['replays'])
    assert {(rp['length'], rp['correct']) for rp in body['replays']} == {(256, 301), (1024, 0)}


def test_decay_summary_has_sixteen_rows_with_four_layers_each():
    r = client.get('/api/length/decay')
    assert r.status_code == 200, r.text
    body = r.json()
    saved = json.loads(DECAY.read_text(encoding='utf-8'))['results']
    assert len(body['rows']) == 16 == len(saved)
    assert all(len(row['layers']) == 4 for row in body['rows'])
    assert {row['filler_tokens'] for row in body['rows']} == {192, 960}
    assert len(body['histogram']['records']) == 2
    assert all(sum(rec['counts']) == 2048 for rec in body['histogram']['records'])
    assert 'do not alone imply state retention' in body['protocol']['interpretation_guardrail']


def test_run_at_zero_filler_has_no_decay_and_a_valid_example():
    r = client.post('/api/length/run', json=dict(init='default', variant='scalar', seed=42, filler=0, examples=2))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['length'] == 64
    assert body['total'] == 8
    assert body['audit_passed'] is True
    assert len(body['example']['tokens']) == 64
    assert body['survival']['steps'] == [0]
    assert all(layer['log10_p50'] == [0.0] for layer in body['survival']['layers'])
    assert len(body['at_filler']) == 4


def test_run_reproduces_saved_diagnostic_at_192_filler():
    # Same data as the saved diagnostic (16 examples, seed offset 1950000 + seed) and same insertion rule.
    r = client.post('/api/length/run', json=dict(init='mean-decay', variant='scalar', seed=44, filler=192, examples=16))
    assert r.status_code == 200, r.text
    body = r.json()
    saved = next(row for row in json.loads(DECAY.read_text(encoding='utf-8'))['results']
                 if row['initialization'] == 'calibrated' and row['variant'] == 'scalar'
                 and row['seed'] == 44 and row['length'] == 256)
    assert body['correct'] == saved['metrics']['correct'] == 64
    assert body['total'] == saved['metrics']['total'] == 64
    assert body['saved'] is not None
    for live_layer, saved_layer in zip(body['at_filler'], saved['layers']):
        assert abs(live_layer['product_p50'] - saved_layer['product_p50']) < 1e-3 * max(1.0, saved_layer['product_p50'])


def test_run_survival_curve_is_monotone_and_has_one_point_per_filler_count():
    r = client.post('/api/length/run', json=dict(init='default', variant='channel-wise', seed=44, filler=64, examples=4))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['survival']['steps'] == list(range(65))
    for layer in body['survival']['layers']:
        assert len(layer['log10_p50']) == 65
        assert all(b <= a + 1e-9 for a, b in zip(layer['log10_p50'], layer['log10_p50'][1:]))


def test_run_rejects_out_of_range_requests():
    base = dict(init='default', variant='scalar', seed=42)
    assert client.post('/api/length/run', json={**base, 'filler': 961}).status_code == 422
    assert client.post('/api/length/run', json={**base, 'filler': -1}).status_code == 422
    assert client.post('/api/length/run', json={**base, 'filler': 10, 'examples': 0}).status_code == 422
    assert client.post('/api/length/run', json={**base, 'filler': 10, 'examples': 17}).status_code == 422
    assert client.post('/api/length/run', json={**base, 'seed': 7, 'filler': 10}).status_code == 422
