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


def test_summary_matches_saved_reports():
    body = client.get('/api/decision/summary').json()
    short = json.loads((ROOT / 'result.json').read_text(encoding='utf-8'))
    longer = json.loads((ROOT / 'result (1).json').read_text(encoding='utf-8'))

    assert body['question']['labels'] == short['labels']
    assert len(body['question']['labels']) == 5
    assert [run['id'] for run in body['runs']] == ['short', 'longer']
    first = body['runs'][0]['variants'][0]
    assert first['variant'] == 'scalar'
    assert first['first_train_loss'] == short['results'][0]['first_train_loss']
    assert body['runs'][1]['variants'][1]['last_train_loss'] == longer['results'][1]['last_train_loss']
    assert body['runs'][1]['steps'] == 50
    assert body['runs'][1]['variants'][0]['timed_steps'] == 50


def test_summary_reports_what_is_missing():
    body = client.get('/api/decision/summary').json()
    assert body['pick_comparison_recorded'] is False
    assert body['pre_training_validation_recorded'] is False
    assert body['fine_tuned_weights_saved'] is False
    assert all(p.startswith('output/tinystories/30M/') for p in body['base_checkpoints_local'])
