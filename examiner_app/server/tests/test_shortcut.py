import sys
import time
from pathlib import Path

import torch
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app, roles_for as app_roles_for  # noqa: E402
import shortcut  # noqa: E402
from recipe_model import RecipeGDN  # noqa: E402

client = TestClient(app)

EVAL = dict(init='default', variant='scalar', seed=44, layout='block_first', rewrites=3, distance=8,
            examples=32, example_seed=0, patch='none')


def evaluate(**overrides):
    return client.post('/api/shortcut/evaluate', json={**EVAL, **overrides})


def test_models_returns_twelve_local_overwrite_models():
    r = client.get('/api/shortcut/models')
    assert r.status_code == 200
    models = r.json()
    assert len(models) == 12
    assert {m['init'] for m in models} == {'default', 'mean-decay'}
    assert {m['variant'] for m in models} == {'scalar', 'channel-wise'}
    assert {m['seed'] for m in models} == {42, 43, 44}
    assert all(m['checkpoint'].endswith('latest_checkpoint.pt') for m in models)


def test_summary_has_five_hypotheses_and_eighteen_models_twelve_local():
    r = client.get('/api/shortcut/summary')
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body['hypotheses']) == {'H1', 'H2', 'H3', 'H4', 'H5'}
    assert all(isinstance(h['confirmed'], bool) for h in body['hypotheses'].values())
    assert len(body['per_model']) == 18
    assert sum(row['local'] for row in body['per_model']) == 12
    assert {'patches', 'stress_first_sign', 'provenance', 'rule_agreement'} <= set(body)
    assert body['provenance']['files']


def test_summary_hypotheses_have_readout_fields_computed_from_observed():
    body = client.get('/api/shortcut/summary').json()
    hyp = body['hypotheses']
    for label, h in hyp.items():
        assert isinstance(h['observed_text'], str) and h['observed_text'].endswith('.'), label
        assert isinstance(h['threshold'], str), label
        assert isinstance(h['observed'], dict), label          # raw object kept
        assert 0 <= h['observed_value'] <= 1, label
        assert 0 <= h['threshold_value'] <= 1, label
    h1 = hyp['H1']
    assert h1['observed_value'] == h1['observed']['models_showing'] / h1['observed']['of']
    assert h1['observed_text'] == f"{h1['observed']['models_showing']} of {h1['observed']['of']} models show the shortcut."
    assert h1['threshold_value'] == 0.5
    counts = hyp['H2']['observed']['error_counts']
    total = sum(counts.values())
    assert hyp['H2']['observed_value'] == counts['first_rewrite'] / total
    assert hyp['H2']['observed_text'].startswith(f"{counts['first_rewrite'] / total * 100:.1f}% of block-first errors")
    assert f"({counts['first_rewrite']:,} of {total:,})" in hyp['H2']['observed_text']
    assert hyp['H2']['threshold_value'] == 0.8
    h3 = hyp['H3']['observed']
    assert hyp['H3']['threshold_value'] == (h3['of'] - 1) / h3['of']
    assert hyp['H3']['observed_value'] == h3['models_at_least_95'] / h3['of']
    h4 = hyp['H4']
    assert h4['observed_value'] == min(v['models_within_10_points'] / v['of'] for v in h4['observed']['by_layout'].values())
    assert h4['threshold_value'] == 0.5
    h5 = hyp['H5']['observed']
    assert hyp['H5']['observed_value'] == min(v['deficit_persists'] / v['of'] for v in h5.values())
    assert hyp['H5']['observed_text'].startswith('The gap persists at distance 12 in')


def test_evaluate_default_scalar_44_block_first_shows_shortcut():
    r = evaluate()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['accuracy_latest'] < 0.5
    assert body['first_rewrite_rate'] > 0.8
    assert body['rule_agreement']['opening_run_second'] > 0.8
    assert len(body['examples']) == 3


def test_evaluate_default_scalar_43_block_first_is_correct():
    r = evaluate(seed=43)
    assert r.status_code == 200, r.text
    assert r.json()['accuracy_latest'] > 0.9


def test_patch_key_align_latest_runs_and_leaves_cached_model_unchanged():
    before = evaluate().json()
    r = evaluate(patch='key_align_latest')
    assert r.status_code == 200, r.text
    patched = r.json()
    assert patched['patch'] == 'key_align_latest'
    assert patched['attribution'] is not None and len(patched['attribution']) == 4
    model = shortcut._load(str(shortcut.local_checkpoints()[('default', 'scalar', 44)]), 'scalar', 44)
    assert not any('forward' in m.__dict__ for m in model.modules() if isinstance(m, RecipeGDN))
    after = evaluate().json()
    assert after['accuracy_latest'] == before['accuracy_latest']
    assert after['answer_sources'] == before['answer_sources']
    assert after['slot_histogram'] == before['slot_histogram']


def test_restore_after_patch_gives_identical_predictions():
    """Stock path before and after a patched run must give identical latest answers."""
    shortcut.evaluate('default', 'scalar', 44, 'block_first', 3, 8, 16, 0, 'none')
    first = shortcut.evaluate('default', 'scalar', 44, 'block_first', 3, 8, 16, 0, 'none')
    shortcut.evaluate('default', 'scalar', 44, 'block_first', 3, 8, 16, 0, 'beta_zero_rewrite1')
    second = shortcut.evaluate('default', 'scalar', 44, 'block_first', 3, 8, 16, 0, 'none')
    assert [e['prediction'] for e in first['examples']] == [e['prediction'] for e in second['examples']]
    assert first['answer_sources'] == second['answer_sources']


def test_bad_layout_is_422():
    assert evaluate(layout='initial_then_run', rewrites=1).status_code == 422
    assert evaluate(layout='not_a_layout').status_code == 422


def test_unknown_patch_and_bad_model_inputs_are_422():
    assert evaluate(patch='rotate_everything').status_code == 422
    assert evaluate(init='joint').status_code == 422
    assert evaluate(examples=65).status_code == 422
    assert evaluate(rewrites=5).status_code == 422


def test_timing_64_examples_warm_under_four_seconds():
    evaluate(examples=64)   # warm: sampling cache and model load
    start = time.perf_counter()
    r = evaluate(examples=64)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200, r.text
    assert elapsed < 4, f'evaluate took {elapsed:.2f}s'


def test_slot_histogram_and_answer_sources_are_consistent():
    body = evaluate().json()
    wrong = round((1 - body['accuracy_latest']) * body['n_examples'])
    assert sum(row['count'] for row in body['slot_histogram']) + body['unmatched_wrong'] == wrong
    assert abs(sum(body['answer_sources'].values()) - 1) < 1e-9
    assert all(row['tokens_before_query'] >= 0 for row in body['slot_histogram'])


def test_example_roles_match_layout_fix_roles():
    body = evaluate(examples=1).json()
    example = body['examples'][0]
    records = list(shortcut._sample(0, 8, 3, 'block_first', 1))
    assert example['roles'] == app_roles_for(records[0], example['tokens'])
    assert len(example['roles']) == len(example['tokens'])


def test_pick_histogram_default_scalar_44_block_first_points_at_first_rewrite():
    body = evaluate(examples=64).json()
    hist = body['pick_histogram']
    total = sum(row['count'] for row in hist)
    assert total == body['n_examples'] == 64
    slot2 = next(row for row in hist if row['slot'] == 2)
    assert slot2['count'] / total > 0.8
    gold_slot = next(row for row in hist if row['is_latest_target'])
    assert gold_slot['slot'] == 4
    assert hist[-1] == {'slot': None, 'count': hist[-1]['count']}


def test_pick_histogram_default_scalar_43_block_first_points_at_newest():
    body = evaluate(seed=43, examples=64).json()
    hist = body['pick_histogram']
    slot4 = next(row for row in hist if row['slot'] == 4)
    assert slot4['count'] / body['n_examples'] > 0.8
    assert slot4['is_latest_target'] and slot4['is_target']


def test_example_writes_are_memory_in_write_order_with_target_flags():
    body = evaluate(examples=8, show_examples=8).json()
    assert len(body['examples']) == 8
    example = body['examples'][0]
    writes = example['writes']
    assert len(writes) == 7
    assert [w['slot'] for w in writes] == list(range(1, 8))
    assert [w['is_target'] for w in writes] == [True, True, True, True, False, False, False]
    assert [w['is_latest_target'] for w in writes] == [False, False, False, True, False, False, False]
    assert [w['value'] for w in writes[:4]] == example['write_values']
    assert all(w['key'] == example['question_key'] for w in writes[:4])
    assert example['gold'] == example['write_values'][-1]
    assert example['gold_slot'] == 4
    assert example['tokens_between_last_write_and_question'] >= 0


def test_example_picked_slot_matches_prediction_and_null_when_absent():
    body = evaluate(examples=32, show_examples=8).json()
    for ex in body['examples']:
        if ex['picked_slot'] is None:
            assert all(w['value'] != ex['prediction'] for w in ex['writes'])
        else:
            assert ex['writes'][ex['picked_slot'] - 1]['value'] == ex['prediction']
            later = [w for w in ex['writes'][ex['picked_slot']:] if w['value'] == ex['prediction']]
            assert later == []


def test_show_examples_is_validated_and_defaults_to_three():
    assert len(evaluate().json()['examples']) == 3
    assert evaluate(show_examples=0).status_code == 422
    assert evaluate(show_examples=9).status_code == 422
