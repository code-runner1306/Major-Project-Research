import json
import statistics
import sys
from pathlib import Path

from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[1]
ROOT = SERVER.parents[1]
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

from app import app  # noqa: E402

client = TestClient(app)
A6000 = ROOT / 'output' / 'a6000_recipe_timing_20261116_083756_121'
T4 = ROOT / 'research_results' / 'published_recipe_timing_20261116'


def test_parameters_match_recipe_constructors():
    r = client.get('/api/cost/parameters')
    assert r.status_code == 200
    p = r.json()
    scalar = p['architectures']['scalar']
    channel = p['architectures']['channel-wise']
    assert scalar['total_parameters'] == 1_585_416
    assert channel['total_parameters'] == 1_651_456
    assert p['delta']['parameters'] == 66_040
    assert round(p['delta']['percent_of_scalar'], 2) == 4.17
    assert p['delta']['per_layer_parameters'] == 16_510
    assert [layer['gate_width'] for layer in scalar['layers']] == [1, 1, 1, 1]
    assert [layer['gate_width'] for layer in channel['layers']] == [128, 128, 128, 128]


def test_recurrent_state_is_the_same_size_in_both_models():
    p = client.get('/api/cost/parameters').json()
    assert p['delta']['state_bytes_difference'] == 0
    assert p['architectures']['scalar']['state_bytes_fp32'] == 524_288
    assert p['delta']['extra_gate_macs_per_token'] == 4 * (128 * 128 - 128)


def test_summary_has_provenance_and_both_gpus_only():
    s = client.get('/api/cost/summary').json()
    assert s['provenance']['file'].endswith('cost_export.json')
    hardware = {h['id'] for h in s['hardware']}
    assert hardware == {'a6000', 't4'}
    assert {r['hardware'] for r in s['step_timing']} == {'a6000', 't4'}
    assert all(r['parameters'] >= 1_585_416 for r in s['step_timing'])  # no CPU smoke model
    assert {e['source'].split('/')[0] for e in s['excluded']} >= {'output'}


def test_a6000_means_match_the_saved_step_lists():
    s = client.get('/api/cost/summary').json()
    run = next(r for r in s['step_timing']
               if r['hardware'] == 'a6000' and r['family'] == 'mqar' and r['mode'] == 'parallel' and r['variant'] == 'scalar')
    raw = json.loads((A6000 / 'mqar' / 'parallel' / 'scalar' / 'summary.json').read_text(encoding='utf-8'))
    assert run['accepted_optimizer_steps'] == len(raw['optimizer_step_seconds']) == 50
    assert abs(run['mean_step_seconds'] - statistics.mean(raw['optimizer_step_seconds'])) < 1e-9
    assert run['parameters'] == raw['parameters'] == 1_585_416


def test_t4_runs_are_recorded_as_mode_not_recorded():
    s = client.get('/api/cost/summary').json()
    t4 = [r for r in s['step_timing'] if r['hardware'] == 't4']
    assert len(t4) == 4
    assert all(r['mode'] == 'not recorded' for r in t4)
    raw = json.loads((T4 / 'mqar' / 'channel-wise' / 'summary.json').read_text(encoding='utf-8'))
    run = next(r for r in t4 if r['family'] == 'mqar' and r['variant'] == 'channel-wise')
    assert abs(run['mean_step_seconds'] - statistics.mean(raw['optimizer_step_seconds'])) < 1e-9


def test_projection_pairs_match_the_benchmark_report():
    s = client.get('/api/cost/summary').json()
    report = json.loads((A6000 / 'benchmark_report.json').read_text(encoding='utf-8'))
    pair = next(p for p in s['projection']['pairs'] if p['family'] == 'mqar' and p['mode'] == 'parallel')
    assert abs(pair['estimated_pair_seconds'] - report['comparisons']['mqar']['parallel']['estimated_pair_training_seconds']) < 1e-6
    assert s['projection']['recommended_mode'] == {'mqar': 'parallel', 'overwrite': 'parallel'}


def test_energy_and_memory_gaps_are_stated_for_the_recipe():
    s = client.get('/api/cost/summary').json()
    assert 'recipe_energy' in s['not_recorded'] and 'recipe_memory' in s['not_recorded']
    assert all(r['family'] in ('mqar', 'overwrite') for r in s['other_models']['full_research_10m'])
    tiny = s['other_models']['tinystories_10m']
    assert {r['variant'] for r in tiny} == {'scalar', 'channel-wise'}
    assert all(r['device'] == 'Tesla T4' for r in tiny)
