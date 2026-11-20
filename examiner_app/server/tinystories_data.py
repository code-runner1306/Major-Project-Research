"""TinyStories language models: saved results, the 30M checkpoints (live, CPU) and the gate readout.

Reads only: output/tinystories/{10M,30M}/<variant>/<run>/metrics.json and manifest.json, and the Das table in
decay_design_results.md. output/tinystories/55M is a copy of the 30M runs and is never read as a result.
The Das numbers are kept separate and labelled with a trailing '*' by the caller.
"""
import json
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dataset_generators as datasets  # noqa: E402
from model import GatedDeltaNetLM  # noqa: E402
from config import ModelConfig  # noqa: E402

RUN_ROOT = ROOT / 'output' / 'tinystories'
MEASURED_SIZES = ('10M', '30M')          # 55M is a copy of the 30M folder, never a result
VARIANT_DIRS = {'scalar': 'scalar', 'channel-wise': 'channel-wise'}
DAS_FILE = ROOT / 'decay_design_results.md'
LIVE_SIZE = '30M'
LIVE_SEED = 42
PROMPT_TOKENS_MAX = 128
NEW_TOKENS_MAX = 40
TEXT_CHARS_MAX = 600
EOT = datasets.TINYSTORIES_EOT
CURVE_POINTS = 100

# Serialises every forward pass on the cached models (same idea as app.py's _lock).
MODEL_LOCK = threading.Lock()


def _smooth(values, points=CURVE_POINTS):
    """Block means: 1,000 training steps become 100 points."""
    if not values:
        return []
    block = max(1, len(values) // points)
    out = []
    for start in range(0, len(values), block):
        chunk = values[start:start + block]
        out.append(round(sum(chunk) / len(chunk), 4))
    return out


def _run_record(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    manifest = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))
    training = data['training']
    fv = data['final_validation']
    layers = (data.get('gate_diagnostics') or {}).get('layers') or {}
    gate = {}
    for index, stats in sorted(layers.items(), key=lambda kv: int(kv[0])):
        gate[index] = dict(
            count=stats.get('count'),
            mean=stats.get('mean'),
            half_life_median=(stats.get('half_life_quantiles') or {}).get('0.5'),
            channel_diversity=stats.get('channel_diversity'),
        )
    return dict(
        size=data['scale'],
        variant=data['variant'],
        seed=data['seed'],
        params=data['parameter']['parameter_count'],
        steps=training.get('fixed_optimizer_steps'),
        tokens=training.get('processed_tokens'),
        val_loss=fv['loss'],
        val_perplexity=fv['perplexity'],
        train_curve=_smooth(training['train_losses']),
        gate_layers=gate,
        completed=data['final_checkpoint']['completed_steps'],
        manifest_model=dict(
            d_model=manifest['model']['d_model'], n_layer=manifest['model']['n_layer'],
            n_head=manifest['model']['n_head'], d_ff=manifest['model']['d_ff'],
            attention_layout=manifest['model']['attention_layout'],
        ),
        run_dir=path.parent.name,
    )


@lru_cache(maxsize=1)
def measured_runs():
    """Every measured run folder under 10M and 30M, read once."""
    runs = []
    for size in MEASURED_SIZES:
        for variant, folder in VARIANT_DIRS.items():
            for path in sorted((RUN_ROOT / size / folder).glob('seed_*/metrics.json')):
                runs.append(_run_record(path))
    return tuple(runs)


def attention_layers(n_layer, layout):
    """1-based layers that are full softmax attention: the rule in model.py GatedDeltaNetBlock (layer_idx % 4 == 3, hybrid only)."""
    if layout != 'hybrid':
        return []
    return [i + 1 for i in range(n_layer) if i % 4 == 3]


def _mean(values):
    return sum(values) / len(values) if values else None


def _std(values):
    if len(values) < 2:
        return None
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def _num(text):
    return float(text.replace('**', '').strip())


@lru_cache(maxsize=1)
def das_rows():
    """The Das et al. table rows (sections 3.1 to 3.3), parsed from decay_design_results.md."""
    text = DAS_FILE.read_text(encoding='utf-8')
    sections = re.split(r'\n### ', text)
    rows = []
    for section in sections:
        heading = section.split('\n', 1)[0]
        match = re.match(r'3\.[123]\s+(\d+)M parameters, TinyStories(\s+\(|$)', heading.strip())
        if not match:
            continue
        size = f'{match.group(1)}M'
        for line in section.splitlines():
            if not line.startswith('|') or ('DeltaNet' not in line and 'KDA' not in line):
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            variant = 'scalar' if 'DeltaNet' in cells[0] else 'channel-wise'
            params = _num(cells[-2])
            loss_cell = cells[-1].replace('**', '')
            nums = re.findall(r'\d+\.\d+', loss_cell)
            value = float(nums[0])
            spread = float(nums[1]) if '±' in loss_cell and len(nums) > 1 else None
            rows.append(dict(size=size, variant=variant, params_m=params, val_loss=value,
                             spread=spread, heading=heading.strip()))
    return tuple(rows)


def _das_seeds_note(size):
    return {'10M': 'mean +- population std over 3 seeds', '55M': 'single seed', '100M': 'mean +- population std over 3 seeds'}.get(size, '')


def summary():
    runs = measured_runs()
    measured = []
    for size in MEASURED_SIZES:
        for variant in VARIANT_DIRS:
            subset = [r for r in runs if r['size'] == size and r['variant'] == variant]
            if not subset:
                continue
            losses = [r['val_loss'] for r in subset]
            arch = subset[0]['manifest_model']
            measured.append(dict(
                size=size, variant=variant, params=subset[0]['params'],
                arch=dict(arch, attention_layers=attention_layers(arch['n_layer'], arch['attention_layout'])),
                seeds=[r['seed'] for r in subset],
                val_loss_per_seed={str(r['seed']): r['val_loss'] for r in subset},
                val_loss_mean=_mean(losses), val_loss_std=_std(losses),
                steps=subset[0]['steps'], tokens=subset[0]['tokens'],
            ))
    paired = []
    for size in MEASURED_SIZES:
        by = {(r['variant'], r['seed']): r['val_loss'] for r in runs if r['size'] == size}
        for seed in sorted({s for (_, s) in by}):
            if ('scalar', seed) in by and ('channel-wise', seed) in by:
                paired.append(dict(size=size, seed=seed,
                                   channel_minus_scalar=by[('channel-wise', seed)] - by[('scalar', seed)]))
    das = [dict(r, label=f"{r['size']}*", seeds_note=_das_seeds_note(r['size'])) for r in das_rows()]
    curves = [dict(size=r['size'], variant=r['variant'], seed=r['seed'], train_curve=r['train_curve'])
              for r in runs]
    return dict(
        measured=measured,
        paired=paired,
        curves=curves,
        das=das,
        budget=dict(steps=1000, tokens=16_384_000, context=512, batch=8, grad_accum=4),
        das_budget_note='Das et al.: 10M unique tokens, 3 epochs, about 7,300 steps. Not our budget.',
        live=dict(size=LIVE_SIZE, seed=LIVE_SEED),
        provenance=dict(
            measured='output/tinystories/{10M,30M}/*/*/metrics.json',
            das=DAS_FILE.name,
            excluded='output/tinystories/55M (a copy of the 30M runs)',
        ),
    )


def checkpoint_path(variant, seed=LIVE_SEED):
    folder = VARIANT_DIRS[variant]
    hits = sorted((RUN_ROOT / LIVE_SIZE / folder).glob(f'seed_{seed}_*/final_checkpoint.pt'))
    return hits[0] if hits else None


def live_models():
    out = []
    for variant in VARIANT_DIRS:
        p = checkpoint_path(variant)
        out.append(dict(variant=variant, seed=LIVE_SEED, size=LIVE_SIZE, present=p is not None,
                        checkpoint=None if p is None else p.relative_to(ROOT).as_posix()))
    return out


@lru_cache(maxsize=2)
def load_model(variant):
    """30M model (eval, CPU), built from the run's manifest and loaded strictly."""
    path = checkpoint_path(variant)
    if path is None:
        raise FileNotFoundError(f'No {LIVE_SIZE} {variant} checkpoint on disk')
    manifest = json.loads((path.parent / 'manifest.json').read_text(encoding='utf-8'))['model']
    cfg = ModelConfig(name=manifest['name'], d_model=manifest['d_model'], n_layer=manifest['n_layer'],
                      n_head=manifest['n_head'], d_ff=tuple(manifest['d_ff']),
                      attention_layout=manifest['attention_layout'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    net = GatedDeltaNetLM(cfg, variant=variant)
    net.load_state_dict(checkpoint['model_state'], strict=True)
    net.eval()
    return net


def _encode(text):
    tokenizer = datasets._gpt2()
    ids = tokenizer.encode(text, disallowed_special=())
    return ids


def _decode(ids):
    return datasets._gpt2().decode([i for i in ids if i != EOT])


def _check_ready():
    datasets._tinystories_ready()   # raises datasets.Unavailable with a plain message


def continue_text(prompt, max_new_tokens):
    """Greedy continuation for both variants; returns text and token ids per variant."""
    _check_ready()
    ids = _encode(prompt)
    if not ids:
        raise ValueError('the story opening has no tokens')
    ids = ids[-PROMPT_TOKENS_MAX:]
    tensor = torch.tensor([ids], dtype=torch.long)
    result = {}
    with MODEL_LOCK, torch.no_grad():
        for variant in VARIANT_DIRS:
            net = load_model(variant)
            out, _ = net.greedy_decode(tensor, max_new_tokens)
            new = out[0, len(ids):].tolist()
            stop = new.index(EOT) if EOT in new else len(new)
            new = new[:stop]
            result[variant] = dict(text=_decode(new), tokens=new, stopped_at_end=stop < max_new_tokens)
    return dict(prompt_tokens=len(ids), prompt_text=_decode(ids), variants=result)


def gate_readout(variant, text):
    """Real decay gate (alpha) values per Gated DeltaNet layer for every token of the text."""
    _check_ready()
    ids = _encode(text)
    if not ids:
        raise ValueError('the text has no tokens')
    ids = ids[:PROMPT_TOKENS_MAX]
    tensor = torch.tensor([ids], dtype=torch.long)
    net = load_model(variant)
    captured = {}

    def observer(alpha, kind, layer_idx):
        captured[layer_idx] = alpha[0].float()     # [T, H, width]

    with MODEL_LOCK, torch.no_grad():
        for block in net.blocks:
            if hasattr(block.attn, 'set_gate_observer'):
                block.attn.set_gate_observer(observer)
        try:
            net(tensor)
        finally:
            for block in net.blocks:
                if hasattr(block.attn, 'set_gate_observer'):
                    block.attn.set_gate_observer(None)
    layers = []
    for index in sorted(captured):
        a = captured[index]                         # [T, H, W]
        flat = a.reshape(a.shape[0], -1)
        layers.append(dict(
            layer=index + 1,
            mean=[round(float(v), 4) for v in flat.mean(dim=1)],
            min=[round(float(v), 4) for v in flat.min(dim=1).values],
            max=[round(float(v), 4) for v in flat.max(dim=1).values],
        ))
    return dict(
        variant=variant,
        tokens=ids,
        token_text=[datasets._gpt2().decode([t]) for t in ids],
        layers=layers,
        gate_width=captured[next(iter(captured))].shape[-1] if captured else None,
        attention_layers=[i + 1 for i in range(net.model_config.n_layer)
                          if type(net.blocks[i].attn).__name__ == 'CausalSelfAttention'],
        note='Gate values come from Gated DeltaNet layers only; the listed attention layers have no decay gate.',
    )
