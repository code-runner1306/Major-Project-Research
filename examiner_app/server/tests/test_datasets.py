import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402
import dataset_generators as datasets  # noqa: E402

client = TestClient(app)

LIVE = [item['id'] for item in datasets.CATALOGUE if item['live']]
LIVE_SCORED = [gid for gid in LIVE if gid != 'tinystories']   # tinystories is a language-model window: scored = []
QUERY_ROLES = {'query_latest', 'query_unrelated', 'query_key', 'mask'}


def generate(gid, seed=5, params=None):
    return client.post('/api/datasets/generate', json=dict(generator=gid, seed=seed, params=params or {}))


def ok(gid, seed=5, params=None):
    r = generate(gid, seed, params)
    assert r.status_code == 200, r.text
    return r.json()


def test_catalogue_lists_eleven_generators():
    items = client.get('/api/datasets').json()
    assert [item['id'] for item in items] == ['mqar', 'overwrite', 'arm', 'layout', 'length', 'kata', 'anatomy_e1',
                                              'anatomy_e4', 'anatomy_e5', 'zoology_cell', 'tinystories']
    assert {'id', 'name', 'source', 'origin', 'live', 'params'} <= set(items[0])
    tiny = items[-1]
    assert tiny['live'] is True and tiny['origin'] == 'external'
    assert tiny['params'] == {'window': {'type': 'choice', 'choices': [64, 128, 256, 512], 'default': 128}}


def test_catalogue_choices_match_research_code():
    import layout_fix_data
    import zoology_layout_test
    import confirm_block_layout as cbl
    assert datasets.ARM_NAMES == tuple(layout_fix_data.ARMS)
    assert datasets.ZOOLOGY_CELL_NAMES == tuple(name for name, _, _ in zoology_layout_test.CELLS)
    assert set(datasets.LAYOUT_NAMES) == {'general_nonblock', *cbl.special_patterns(2), *cbl.special_patterns(4)}


@pytest.mark.parametrize('gid', LIVE)
def test_tokens_and_roles_have_equal_length(gid):
    body = ok(gid)
    assert len(body['tokens']) == len(body['roles']) > 0
    assert body['generator'] == gid
    assert 1 <= len(body['notes']) <= 5
    assert all(isinstance(n, str) and n for n in body['notes'])


@pytest.mark.parametrize('gid', LIVE_SCORED)
def test_scored_positions_are_query_or_mask_roles(gid):
    body = ok(gid)
    assert body['scored'], 'every live generator scores at least one position'
    for s in body['scored']:
        assert 0 <= s['position'] < len(body['tokens'])
        assert body['roles'][s['position']] in QUERY_ROLES
        assert isinstance(s['answer'], int) and isinstance(s['kind'], str)


def test_overwrite_is_deterministic_for_a_seed():
    params = (('distance', 8), ('rewrites', 1))
    first = datasets._build.__wrapped__('overwrite', 5, params)
    second = datasets._build.__wrapped__('overwrite', 5, params)
    assert first['tokens'] == second['tokens'] and first['roles'] == second['roles']
    assert first['scored'] == second['scored']
    assert ok('overwrite', 5) == ok('overwrite', 5)


def test_overwrite_matches_retrieval_labels_and_answers():
    from config import RetrievalConfig
    from retrieval import RetrievalDataset
    cfg = RetrievalConfig(vocabulary_size=8192, sequence_lengths=(65,), key_counts=(4,), query_distances=(8,),
                          overwrite_counts=(1,), generator_version='mqar-overwrite-v4')
    record = RetrievalDataset(task='overwrite', examples=1, config=cfg, seed=5, split='train').example(0)
    body = ok('overwrite', 5)
    expected = sorted((a.label_position, int(record.labels[a.label_position]), a.kind) for a in record.answers)
    assert [(s['position'], s['answer'], s['kind']) for s in body['scored']] == expected
    latest = next(a for a in record.answers if a.kind == 'latest')
    assert any(str(int(record.labels[latest.label_position])) in note for note in body['notes'])
    assert body['roles'][latest.label_position - 1] == 'marker'
    assert body['roles'][latest.label_position] == 'query_latest'


@pytest.mark.parametrize('arm', datasets.ARM_NAMES)
def test_arm_answers_match_build_arm_data(arm):
    import layout_fix_data
    data = layout_fix_data.build_arm_data(arm, 1, 5)
    body = ok('arm', 5, {'arm': arm})
    labels = data.labels[0]
    for s in body['scored']:
        assert s['answer'] == int(labels[s['position']])
    assert sorted(s['position'] for s in body['scored']) == sorted(
        int(p) for p in (labels != -100).nonzero().flatten().tolist())


def test_layout_block_first_rewrites_three_has_pattern():
    body = ok('layout', 5, {'layout': 'block_first', 'rewrites': 3})
    assert body['meta']['pattern'] == 'TTTTUUU'
    assert body['meta']['layout'] == 'block_first'


def test_layout_not_valid_for_rewrites_is_422():
    assert generate('layout', 5, {'layout': 'initial_then_run', 'rewrites': 1}).status_code == 422


def test_mqar_row_zero_note_present_and_answers_are_memory_values():
    from recipe_data import mqar
    body = ok('mqar', 5)
    assert 'row 0' in body['meta']['note'] and '256' in body['meta']['note']
    tokens, labels = mqar(256, 8192, 64, 4, 5)
    assert body['tokens'] == tokens[0].tolist()
    memory = {body['tokens'][i]: body['tokens'][i + 1] for i in range(0, 8, 2)}
    for s in body['scored']:
        assert s['answer'] == labels[0][s['position']].item() == memory[body['tokens'][s['position']]]


@pytest.mark.parametrize('length', [64, 256, 1024])
def test_length_extension_inserts_filler_after_memory(length):
    body = ok('length', 5, {'length': length})
    extra = length - 64
    assert len(body['tokens']) == length
    assert body['roles'][8:8 + extra] == ['filler'] * extra
    assert body['tokens'][8:8 + extra] == [0] * extra
    assert body['roles'][:8] == ['key', 'value'] * 4
    base = ok('mqar', 5)
    assert body['tokens'][:8] == base['tokens'][:8]
    assert [s['answer'] for s in body['scored']] == [s['answer'] for s in base['scored']]


def test_kata_k4_passes2_length_and_scored():
    import kata_overwrite_data
    body = ok('kata', 5, {'keys': 4, 'passes': 2})
    assert len(body['tokens']) == 24
    assert len(body['scored']) == 4
    tokens, labels, writes = kata_overwrite_data.generate(1, 4, 2, 5, 'train', partial_last=False)
    assert [s['answer'] for s in body['scored']] == [int(labels[0][s['position']]) for s in body['scored']]
    for s in body['scored']:
        key = body['tokens'][s['position']]
        values = [body['tokens'][i + 1] for i in range(0, 16, 2) if body['tokens'][i] == key]
        assert s['answer'] == values[-1]
    assert body['tokens'][1::2][-4:] == [0, 0, 0, 0]


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_anatomy_e1_roles_match_tensors(seed):
    import torch
    import sys as _sys
    _sys.path.append(str(SERVER.parents[1] / 'third_party' / 'dualgoose'))
    from dualgoose import toy_data
    keys = 8
    batch = toy_data.make_mqar_batch(1, 2 * keys + 8, 64, 63, pairs=keys, query_layout='after_table',
                                     generator=torch.Generator().manual_seed(seed))
    body = ok('anatomy_e1', seed, {'keys': keys})
    clean = batch.clean[0].tolist()
    assert body['tokens'] == batch.corrupted[0].tolist()
    assert body['roles'][:2 * keys:2] == ['key'] * keys and body['roles'][1:2 * keys:2] == ['value'] * keys
    assert all(5 <= body['tokens'][i] < 5 + keys for i in range(0, 2 * keys, 2))
    qk = clean[2 * keys]
    assert body['roles'][2 * keys] == 'query_key' and body['tokens'][2 * keys] == qk
    assert body['roles'][2 * keys + 1] == 'mask' and body['tokens'][2 * keys + 1] == 63
    assert body['scored'] == [dict(position=2 * keys + 1, answer=clean[2 * keys + 1], kind='masked_value')]
    assert body['roles'][2 * keys + 2:] == ['noise'] * 6
    table = {clean[i]: clean[i + 1] for i in range(0, 2 * keys, 2)}
    assert table[qk] == clean[2 * keys + 1]


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('gid,collision', [('anatomy_e4', False), ('anatomy_e5', True)])
def test_anatomy_long_roles_match_tensors(gid, collision, seed):
    import torch
    import sys as _sys
    _sys.path.append(str(SERVER.parents[1] / 'third_party' / 'dualgoose'))
    from dualgoose import toy_data
    batch = toy_data.make_long_retrieval_batch(1, 64, 64, 63, pairs=4, n_queries=4, gap=None, collision=collision,
                                               generator=torch.Generator().manual_seed(seed))
    body = ok(gid, seed)
    clean = batch.clean[0].tolist()
    assert body['tokens'] == batch.corrupted[0].tolist()
    assert [body['tokens'][i] for i in range(0, 8, 2)] == [5, 6, 7, 8]
    assert body['roles'][:8] == ['key', 'value'] * 4
    assert all(9 <= body['tokens'][i] <= 62 for i in range(8, 56) if body['roles'][i] in ('noise', 'decoy_value'))
    query_positions = [i for i, r in enumerate(body['roles']) if r == 'query_key']
    assert query_positions == [56, 58, 60, 62]
    assert all(5 <= body['tokens'][i] <= 8 for i in query_positions)
    assert [s['answer'] for s in body['scored']] == [clean[p] for p in (57, 59, 61, 63)]
    for q in query_positions:
        assert clean[q + 1] == clean[2 * (clean[q] - 5) + 1]
    if collision:
        assert body['roles'][8:56:2] == ['decoy_key'] * 24
        assert body['roles'][9:56:2] == ['decoy_value'] * 24
        assert all(5 <= body['tokens'][i] <= 8 for i in range(8, 56, 2))
    else:
        assert body['roles'][8:56] == ['noise'] * 48
        assert all(body['tokens'][i] not in (5, 6, 7, 8) for i in range(8, 56))


def test_anatomy_e5_has_decoy_keys():
    body = ok('anatomy_e5', 3)
    assert body['roles'].count('decoy_key') >= 1


def test_zoology_baseline_cell_is_upstream_mqar_row_zero():
    body = ok('zoology_cell', 5, {'cell': 'zoology_baseline'})
    base = ok('mqar', 5)
    assert body['tokens'] == base['tokens']
    assert body['scored'] == base['scored']


def test_zoology_shifted_cell_places_memory_after_prefix():
    body = ok('zoology_cell', 5, {'cell': 'shift_4'})
    assert body['roles'][:4] == ['filler'] * 4
    assert body['roles'][4:12] == ['key', 'value'] * 4
    assert all(body['roles'][s['position']] == 'query_key' for s in body['scored'])


@pytest.mark.parametrize('window', [64, 128, 256, 512])
def test_tinystories_window_tokens_roles_and_text(window):
    body = ok('tinystories', 5, {'window': window})
    n = len(body['tokens'])
    assert n == len(body['roles']) == len(body['token_text']) <= window
    assert body['scored'] == []
    assert set(body['roles']) <= {'text', 'eos'}
    assert body['meta']['split'] == 'validation' and body['meta']['window'] == window
    assert body['meta']['story_index'] == 5 and body['meta']['file'].endswith('.parquet')
    story_len = body['meta']['num_story_tokens']
    if story_len < window:
        assert n == story_len + 1
        assert body['tokens'][-1] == 50256 and body['roles'][-1] == 'eos'
        assert body['roles'].count('eos') == 1
    else:
        assert n == window and 'eos' not in body['roles']
    assert all(isinstance(t, str) for t in body['token_text'])
    assert 0 < len(body['text']) <= 803 and isinstance(body['text'], str)
    assert 1 <= len(body['notes']) <= 5


def test_tinystories_is_deterministic_and_seed_picks_story_modulo_rows():
    first = ok('tinystories', 5)
    assert first == ok('tinystories', 5)
    assert first == ok('tinystories', 5 + 21990)          # story = seed % 21990
    assert first['meta']['story_index'] == 5


def test_tinystories_tokens_match_dataset_packing():
    import pyarrow.parquet as pq
    texts = pq.read_table(datasets.TINYSTORIES_PARQUET, columns=['text']).column('text').to_pylist()
    assert len(texts) == 21990
    enc = datasets._gpt2()
    full = enc.encode(texts[7], allowed_special='all') + [enc.eot_token]   # as dataset.TinyStoriesDataset
    body = ok('tinystories', 7, {'window': 512})
    assert body['tokens'] == full[:512]


def test_tinystories_missing_parquet_is_409(monkeypatch, tmp_path):
    missing = tmp_path / 'validation-missing.parquet'
    monkeypatch.setattr(datasets, 'TINYSTORIES_PARQUET', missing)
    r = generate('tinystories', 5)
    assert r.status_code == 409
    assert 'validation-missing.parquet' in r.json()['detail'] and 'missing' in r.json()['detail']


@pytest.mark.parametrize('gid,params', [
    ('overwrite', {'distance': 40}),
    ('overwrite', {'rewrites': 0}),
    ('overwrite', {'foo': 1}),
    ('kata', {'keys': 5}),
    ('kata', {'passes': True}),
    ('arm', {'arm': 'A9_nope'}),
    ('length', {'length': 128}),
    ('zoology_cell', {'cell': 'nope'}),
    ('layout', {'layout': 'not_a_layout'}),
    ('nonexistent_generator', {}),
])
def test_invalid_params_are_422(gid, params):
    assert generate(gid, 5, params).status_code == 422


def test_negative_seed_is_422():
    assert generate('mqar', -1).status_code == 422
    assert generate('mqar', 2 ** 32).status_code == 422


@pytest.mark.parametrize('gid', LIVE)
def test_warm_call_under_one_second(gid):
    ok(gid, 10)                       # warm-up: imports and first build
    start = time.perf_counter()
    ok(gid, 11)                       # a fresh seed, so nothing is served from the cache
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f'{gid} took {elapsed:.2f}s'
