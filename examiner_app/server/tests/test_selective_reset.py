import sys
from pathlib import Path

from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402
import selective_reset_data as data  # noqa: E402

client = TestClient(app)


def test_summary_reports_four_saved_runs_with_the_recorded_parameter_counts():
    r = client.get('/api/selective-reset/summary')
    assert r.status_code == 200, r.text
    body = r.json()
    runs = {(run['decay'], run['seed']): run for run in body['runs']}
    assert set(runs) == {('channel-wise', 42), ('channel-wise', 43), ('scalar', 42), ('scalar', 43)}
    assert runs[('channel-wise', 42)]['parameters'] == 49732
    assert runs[('scalar', 42)]['parameters'] == 45640
    assert all([row['load'] for row in run['by_load']] == [1, 2, 4, 8] for run in body['runs'])


def test_no_run_passes_the_joint_test_and_survivor_stays_below_guess_rate():
    body = client.get('/api/selective-reset/summary').json()
    assert body['max_joint'] <= 0.01
    assert body['max_survivor'] < body['task']['guess_rate']
    assert body['min_reset_target'] == 1.0


def test_summary_marks_the_one_checkpoint_as_live():
    body = client.get('/api/selective-reset/summary').json()
    assert body['live'] is True
    assert body['checkpoints'] == [{'decay': 'channel-wise', 'seed': 42, 'file': 'gated_deltanet_42.zip'}]


def test_checkpoint_model_has_the_recorded_parameter_count():
    model = data.load_model('gated_deltanet', 42)
    assert sum(p.numel() for p in model.parameters()) == 49732


def test_try_returns_one_episode_with_64_tokens_and_a_clear_expected_answer():
    r = client.post('/api/selective-reset/try', json={'load': 2, 'reset': True, 'query': 'target', 'seed': 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body['tokens']) == 64
    assert body['expected_label'] == 'empty'
    assert len(body['top3']) == 3 and abs(sum(t['prob'] for t in body['top3'])) <= 1.0


def test_try_rejects_a_memory_load_outside_the_pilot_loads():
    r = client.post('/api/selective-reset/try', json={'load': 3, 'reset': False, 'query': 'survivor', 'seed': 1})
    assert r.status_code == 422


def test_profile_shares_sum_to_one_for_every_cell():
    body = client.get('/api/selective-reset/profile').json()
    assert body['kinds'][0] == 'correct'
    for load, cells in body['by_load'].items():
        assert set(cells) == {f'{mode}_{q}' for mode in ('reset', 'noop')
                              for q in ('target', 'survivor', 'rewrite', 'absent')}, load
        for cell in cells.values():
            assert abs(sum(cell.values()) - 1.0) < 1e-3
