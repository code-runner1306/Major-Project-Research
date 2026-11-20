import json
from pathlib import Path

from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
ROOT = SERVER.parents[1]

from app import app  # noqa: E402

client = TestClient(app)
ATTN = ROOT / 'research_results' / 'attention_control_20261115'
KAGGLE = ROOT / 'research_results' / 'kaggle_seed42_20261114'


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_summary_matches_saved_overwrite_results():
    r = client.get('/api/attention-control/summary')
    assert r.status_code == 200, r.text
    body = r.json()
    att = _read(ATTN / 'overwrite' / 'overwrite' / 'attention' / 'summary.json')['confirmation']['answer_free']
    gdn = _read(ATTN / 'overwrite' / 'overwrite' / 'gdn' / 'summary.json')['confirmation']['answer_free']
    assert body['same_protocol']['overwrite']['attention']['answer_free'] == {
        'correct': att['correct'], 'total': att['total'], 'accuracy': att['exact_query_accuracy']}
    assert body['same_protocol']['overwrite']['gdn']['answer_free']['correct'] == gdn['correct'] == 25
    assert body['same_protocol']['overwrite']['attention']['answer_free']['correct'] == 649


def test_summary_parameters_and_memory_come_from_settings():
    body = client.get('/api/attention-control/summary').json()
    assert body['same_protocol']['overwrite']['attention']['parameters'] == 3174560
    assert body['same_protocol']['overwrite']['gdn']['parameters'] == 3329784
    # 4 heads x 40 x 40 per layer for the recurrent state; attention adds 2 x 4 x 40 numbers per token per layer.
    assert body['memory']['width'] == 160 and body['memory']['layers'] == 6
    assert body['memory']['recurrent_state_numbers_per_layer'] == 4 * 40 * 40
    assert body['memory']['attention_cache_numbers_per_token_per_layer'] == 2 * 4 * 40


def test_kaggle_scalar_and_channel_wise_counts_match_metrics():
    body = client.get('/api/attention-control/summary').json()
    c1 = _read(KAGGLE / 'C1' / 'runs' / 'mqar' / 'C1-scalar-default' / 'seed_42' / 'metrics.json')['retrieval']
    c3 = _read(KAGGLE / 'C3' / 'runs' / 'mqar' / 'C3-channel-default' / 'seed_42' / 'metrics.json')['retrieval']
    assert body['scalar_vs_channel']['mqar']['C1']['answer_free']['total'] == c1['total'] == 4772
    assert body['scalar_vs_channel']['mqar']['C1']['answer_free']['correct'] == c1['correct']
    assert body['scalar_vs_channel']['mqar']['C3']['answer_free']['correct'] == c3['correct'] == 1


def test_paper_gap_is_parsed_from_decay_note():
    body = client.get('/api/attention-control/summary').json()
    assert body['paper_context']['channel_minus_scalar_gap'] == {'10M': -0.0402, '100M': -0.0389}


def test_status_is_results_only_without_attention_checkpoints():
    body = client.get('/api/attention-control/summary').json()
    assert body['status']['live'] is False
    assert body['status']['checkpoints'] == []


def test_curves_have_one_point_per_evaluation():
    body = client.get('/api/attention-control/summary').json()
    curve = body['same_protocol']['overwrite']['attention']['development_curve']
    assert [p['step'] for p in curve] == list(range(200, 2001, 200))


def test_example_tokens_roles_and_queries_are_consistent():
    r = client.get('/api/attention-control/example?index=0')
    assert r.status_code == 200, r.text
    ex = r.json()
    assert len(ex['tokens']) == len(ex['roles']) == ex['token_count']
    assert all(role == 'filler' or isinstance(role, str) for role in ex['roles'])
    assert [q['kind'] for q in ex['queries']] == ['latest', 'unrelated']
    for q in ex['queries']:
        assert ex['roles'][q['query_position']] == 'marker'
        assert ex['roles'][q['key_position']].startswith('query_')
    latest = ex['queries'][0]
    assert ex['tokens'][latest['key_position']] == ex['tokens'][ex['writes']['key_positions'][1]]


def test_example_index_out_of_range_is_rejected():
    assert client.get('/api/attention-control/example?index=512').status_code == 422
