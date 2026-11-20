import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402
import anatomy_data  # noqa: E402

client = TestClient(app)
ROOT = SERVER.parents[1]
REPORT = ROOT / 'third_party' / 'dualgoose' / 'results' / 'granularity_report.json'


def test_summary_has_three_cells_and_no_checkpoints():
    body = client.get('/api/anatomy/summary').json()
    assert [c['id'] for c in body['cells']] == ['E1', 'E4', 'E5']
    assert body['checkpoints_available'] is False
    assert body['cells'][2]['collision'] is True and body['cells'][1]['collision'] is False


def test_summary_matches_saved_report():
    body = client.get('/api/anatomy/summary').json()
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    for cell in body['cells']:
        ours = {a['decay']: a for a in cell['arms'] if a['source'] == 'ours'}
        saved = report['cells'][cell['id']]
        assert ours['channel-wise']['n'] == saved['channel']['n'] == 10
        assert ours['channel-wise']['mean'] == pytest.approx(saved['channel']['mean'])
        assert ours['scalar']['mean'] == pytest.approx(saved['scalar']['mean'])
        assert ours['channel-wise']['lock_in'] == saved['channel']['lock_in']
        assert ours['scalar']['lock_in'] == saved['scalar']['lock_in']
        assert [s['accuracy'] for s in ours['scalar']['per_seed']] == pytest.approx(saved['scalar']['per_seed'], abs=1e-3)


def test_reproduction_matches_published_reference():
    """Our channel-wise arm reproduces the published numbers (the report's R0 check)."""
    body = client.get('/api/anatomy/summary').json()
    for cell in body['cells']:
        ours = next(a for a in cell['arms'] if a['id'] == 'ours_channel-wise')
        up = next(a for a in cell['arms'] if a['id'] == 'published_armed_gdn')
        assert abs(ours['mean'] - up['mean']) < 0.05
        assert ours['lock_in'] == up['lock_in']   # 10/10 at E1, 6/10 at E4, 3/10 at E5 in both


def test_tests_are_copied_from_report():
    body = client.get('/api/anatomy/summary').json()
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    for cell_id, t in body['tests'].items():
        assert t['permutation_p'] == pytest.approx(report['cells'][cell_id]['permutation_p'])
        assert t['lock_in_fisher_p'] == pytest.approx(report['cells'][cell_id]['lock_in_fisher_p'])
    # E1: the scalar arm is higher (channel minus scalar is negative), and the difference is significant.
    assert body['tests']['E1']['mean_difference'] < 0
    assert body['tests']['E1']['permutation_p'] < 0.05


def test_interventions_present_for_all_cells():
    body = client.get('/api/anatomy/summary').json()
    assert set(body['interventions']) == {'E1', 'E4', 'E5'}
    for cell in body['interventions'].values():
        assert cell['n'] == 10
        assert set(cell['means']) == set(body['intervention_labels'])


def test_runs_filter_and_shape():
    rows = client.get('/api/anatomy/runs', params={'cell': 'E5', 'source': 'ours'}).json()
    assert len(rows) == 20                       # 10 channel-wise + 10 scalar
    assert {r['arm'] for r in rows} == {'armed_gdn', 'armed_gdn_scalar'}
    pub = client.get('/api/anatomy/runs', params={'cell': 'E4', 'source': 'published'}).json()
    assert all(r['source'] == 'published' and r['rebind_accuracy'] is None for r in pub)


def test_example_is_the_real_generator_output():
    import dataset_generators as datasets
    for experiment, gid in (('E4', 'anatomy_e4'), ('E5', 'anatomy_e5')):
        body = client.get('/api/anatomy/example', params={'experiment': experiment, 'seed': 5}).json()
        ref = datasets.generate(gid, 5, {})
        assert body['tokens'] == ref['tokens'] and body['roles'] == ref['roles']
        assert len(body['tokens']) == 64 and len(body['scored']) == 4
        assert [s['position'] for s in body['scored']] == [57, 59, 61, 63]
        assert body['experiment'] == experiment


def test_example_rejects_unknown_experiment():
    assert client.get('/api/anatomy/example', params={'experiment': 'E9', 'seed': 1}).status_code == 422
