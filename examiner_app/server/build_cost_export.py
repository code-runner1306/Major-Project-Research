"""Build server/data/cost_export.json from saved timing, energy and memory artifacts.

Run from examiner_app/server with the repo venv (no torch needed):

    ..\\..\\..\\.venv\\Scripts\\python.exe build_cost_export.py

Reads only. Writes only server/data/cost_export.json. Every number on the Cost page comes from this file
or from the live parameter count in cost_data.py. Hardware is recorded per run and never mixed.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'data' / 'cost_export.json'

# Recipe-model timing on the single RTX A6000 virtual profile (measured; 50 optimizer steps per run,
# projections to 64 epochs = 12,544 steps). The benchmark wrote this as 083756_121.
A6000_DIR = ROOT / 'output' / 'a6000_recipe_timing_20261116_083756_121'
# Recipe-model timing on Kaggle GPUs, saved by run_kaggle_published_recipe.sh (GPU recorded as Tesla T4).
T4_DIR = ROOT / 'research_results' / 'published_recipe_timing_20261116'
# The two other output folders are CPU smoke runs of a tiny configuration (vocabulary and width reduced).
CPU_SMOKE_DIRS = [ROOT / 'output' / 'a6000_recipe_timing_20261116_083251_635',
                  ROOT / 'output' / 'a6000_recipe_timing_20261116_083154_572']
JOINT = ROOT / 'research_results' / 'joint_calibration_runtime_estimate_20261117' / 'estimate.json'
FULL = ROOT / 'output' / 'full_research'
TINY = ROOT / 'output' / 'tinystories'
BATCH = 512  # effective batch: examples per optimizer step


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def run_stats(steps):
    sd = statistics.stdev(steps) if len(steps) > 1 else 0.0
    mean = statistics.mean(steps)
    return {
        'mean_step_seconds': mean,
        'median_step_seconds': statistics.median(steps),
        'min_step_seconds': min(steps),
        'max_step_seconds': max(steps),
        'sd_step_seconds': sd,
        'se_step_seconds': sd / (len(steps) ** 0.5),
        'examples_per_second': BATCH / mean,
    }


def step_timing_runs():
    runs = []
    # A6000: <family>/<mode>/<variant>/summary.json
    for path in sorted(A6000_DIR.glob('*/*/*/summary.json')):
        family, mode, variant = path.parts[-4], path.parts[-3], path.parts[-2]
        s = load(path)
        settings = load(path.parent / 'settings.json')
        steps = s['optimizer_step_seconds']
        runs.append({
            'hardware': 'a6000', 'family': family, 'mode': mode, 'variant': variant,
            'gpu': s['gpu'], 'parameters': s['parameters'], 'accepted_optimizer_steps': s['accepted_optimizer_steps'],
            'effective_batch': settings['effective_batch'], 'length': settings['length'],
            'precision': settings['precision'], 'microbatch': settings['microbatch'],
            **run_stats(steps), 'setup_seconds': s['setup_seconds'], 'validation_seconds': s['validation_seconds'],
            'checkpoint_seconds': s['checkpoint_seconds'], 'wall_seconds': s['wall_seconds'],
            'step_seconds': steps, 'source': rel(path),
        })
    # Kaggle: <family>/<variant>/summary.json. The saved summary does not record whether two workers shared a GPU.
    for path in sorted(T4_DIR.glob('*/*/summary.json')):
        family, variant = path.parts[-3], path.parts[-2]
        s = load(path)
        settings = load(path.parent / 'settings.json')
        steps = s['optimizer_step_seconds']
        runs.append({
            'hardware': 't4', 'family': family, 'mode': 'not recorded', 'variant': variant,
            'gpu': s['gpu'], 'parameters': s['parameters'], 'accepted_optimizer_steps': s['accepted_optimizer_steps'],
            'effective_batch': settings['effective_batch'], 'length': settings['length'],
            'precision': settings['precision'], 'microbatch': settings['microbatch'],
            **run_stats(steps), 'setup_seconds': s['setup_seconds'], 'validation_seconds': s['validation_seconds'],
            'checkpoint_seconds': s['checkpoint_seconds'], 'wall_seconds': s['wall_seconds'],
            'step_seconds': steps, 'source': rel(path),
        })
    return runs


def projections():
    report = load(A6000_DIR / 'benchmark_report.json')
    jobs = []
    for key, job in sorted(report['jobs'].items()):
        family, mode, variant = key.split('/')
        jobs.append({'family': family, 'mode': mode, 'variant': variant,
                     'optimizer_steps': job['optimizer_steps'],
                     'estimated_seconds': job['estimated_seconds'],
                     'conservative_seconds': job['conservative_seconds'],
                     'mean_step_seconds': job['mean_step_seconds']})
    pairs = []
    recommended = {}
    for family, modes in report['comparisons'].items():
        recommended[family] = modes.get('recommended_mode_by_projected_pair_time')
        for mode in ('parallel', 'sequential'):
            m = modes[mode]
            pairs.append({'family': family, 'mode': mode,
                          'estimated_pair_seconds': m['estimated_pair_training_seconds'],
                          'conservative_pair_seconds': m['conservative_pair_training_seconds']})
    return {'source': rel(A6000_DIR / 'benchmark_report.json'), 'optimizer_steps': 12544, 'epochs': 64,
            'jobs': jobs, 'pairs': pairs, 'recommended_mode': recommended}


def joint_measured():
    e = load(JOINT)
    return {
        'source': rel(JOINT), 'hardware': 'a6000',
        'optimizer_steps_per_model': e['optimizer_steps_per_model'], 'epochs': e['epochs'],
        'paired_sessions': e['paired_sessions'],
        'measured_pair_seconds': e['measured_pair_seconds'],
        'note': 'Two-worker sessions on the A6000 profile, measured for a 16-epoch budget. '
                'The joint initialization itself was not implemented or trained.',
    }


def full_research_runs():
    out = []
    for path in sorted(FULL.glob('*/10M/*/*/metrics.json')):
        d = load(path)
        tr = d['training']
        steps = tr['optimizer_step_seconds']
        te = d.get('training_energy') or {}
        tm = (d.get('training_resources') or {}).get('torch_memory') or {}
        decode = d['benchmarks']['context_256']['cached_decode']
        out.append({
            'family': d['family'], 'variant': d['variant'], 'seed': d['seed'],
            'parameters': d['parameter']['parameter_count'],
            'gate_parameter_overhead': d['parameter']['gate_parameter_overhead'],
            'device': (te.get('device') or {}).get('name'),
            'training_wall_seconds': tr['wall_time_seconds'],
            'training_steps': tr['fixed_optimizer_steps'],
            'training_tokens_per_second': tr['training_tokens_per_second'],
            'mean_step_seconds': statistics.mean(steps),
            'peak_allocated_bytes': tm.get('peak_allocated_bytes'),
            'energy_supported': te.get('supported'),
            'energy_joules': te.get('joules'),
            'energy_reason': te.get('reason') or None,
            'decode_ms_per_token_context_256': decode['milliseconds_per_generated_token'],
            'decode_state_bytes': decode['state_memory_bytes'],
            'source': rel(path),
        })
    return out


def tinystories_10m_runs():
    out = []
    for path in sorted(TINY.glob('10M/*/*/metrics.json')):
        d = load(path)
        tr = d['training']
        te = d.get('training_energy') or {}
        out.append({
            'variant': d['variant'], 'seed': d['seed'], 'parameters': d['parameter']['parameter_count'],
            'gate_parameter_overhead': d['parameter']['gate_parameter_overhead'],
            'device': (te.get('device') or {}).get('name'),
            'training_wall_seconds': tr['wall_time_seconds'], 'training_steps': tr['fixed_optimizer_steps'],
            'processed_tokens': tr['processed_tokens'],
            'energy_supported': te.get('supported'), 'energy_joules': te.get('joules'),
            'energy_method': te.get('method'), 'energy_reason': te.get('reason') or None,
            'source': rel(path),
        })
    return out


def main():
    runs = step_timing_runs()
    export = {
        'meta': {
            'generator': 'examiner_app/server/build_cost_export.py',
            'batch_examples_per_step': BATCH,
            'recipe_model': 'RecipeLM, width 128, vocabulary 8192, length 64, 4 layers, one head',
            'hardware_note': 'Each run carries its own hardware label. Comparisons only within one hardware label.',
        },
        'hardware': [
            {'id': 'a6000', 'label': 'NVIDIA RTXA6000-12Q (A6000 profile)', 'kind': 'GPU, measured',
             'source': rel(A6000_DIR)},
            {'id': 't4', 'label': 'Tesla T4 (Kaggle GPU)', 'kind': 'GPU, measured', 'source': rel(T4_DIR)},
        ],
        'excluded': [{'source': rel(p), 'reason': 'CPU smoke run of a tiny configuration (not recipe size, not GPU timing)'}
                     for p in CPU_SMOKE_DIRS] + [
            {'source': 'output/tinystories/30M and 55M', 'reason': 'NVML energy failed; the 55M folder repeats the 30M run records and the parameter count is 30M, so it is not used'}],
        'step_timing': runs,
        'projection': projections(),
        'joint_measured': joint_measured(),
        'other_models': {
            'note': 'Different models from the recipe. Shown only to say what else was measured. Not comparable to the recipe rows.',
            'full_research_10m': full_research_runs(),
            'tinystories_10m': tinystories_10m_runs(),
        },
        'not_recorded': {
            'recipe_energy': 'No energy reading exists for the recipe runs on A6000 or Kaggle T4. benchmark_single_gpu_recipe.py does not call the energy or memory helpers.',
            'recipe_memory': 'No memory reading exists for the recipe runs on A6000 or Kaggle T4 in the saved summaries.',
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(export, indent=2), encoding='utf-8')
    print('wrote', OUT, 'runs', len(runs))


if __name__ == '__main__':
    main()
