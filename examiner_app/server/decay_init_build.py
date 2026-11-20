"""Build server/data/decay_init_gates.json: decay (alpha) values of the 12 local overwrite models.

Run once from examiner_app/server (reproducible; reads only):

    ../../.venv/Scripts/python.exe decay_init_build.py

For every local overwrite checkpoint (default and mean-decay x scalar and channel-wise x seeds 42-44):
  * trained: alpha = exp(-exp(A_log) * softplus(a_proj(x) + dt_bias)) (RecipeGDN.decay, unchanged) averaged over the
    256 calibration rows (the first 256 training rows of that seed; SHA-256 checked against calibration.json) and over
    tokens. One value per channel (channel-wise) or one value (scalar).
  * start: mean-decay only, the achieved per-channel means recorded in calibration.json. The default initialisation
    has no recoverable start (see three_seed_gate_diagnostics REPORT), so its start is null.
  * channel_spread: standard deviation across channels of the per-channel means (0 for scalar).

Nothing is trained or written outside examiner_app/server/data.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SERVER = Path(__file__).resolve().parent
ROOT = SERVER.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SERVER) not in sys.path:
    sys.path.insert(0, str(SERVER))

import analyze_write_gate as awg  # noqa: E402
from recipe_model import RecipeGDN  # noqa: E402
import shortcut  # noqa: E402

ROWS = 256
DIAG = ROOT / 'research_results' / 'three_seed_confirmation_20261117' / 'diagnostic_training_samples'
OUT = SERVER / 'data' / 'decay_init_gates.json'


def _rounded(values, digits=6):
    return [round(float(v), digits) for v in values]


@torch.no_grad()
def trained_alpha(model, tokens):
    """Mean alpha per recurrent layer over all rows and tokens, one value per channel (or one for scalar)."""
    layers = [m for m in model.modules() if isinstance(m, RecipeGDN)]
    out = []
    for layer in layers:
        captured = []
        handle = layer.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach().float()))
        try:
            for offset in range(0, len(tokens), 32):
                model.network.backbone(tokens[offset:offset + 32])
        finally:
            handle.remove()
        alpha = torch.cat([layer.decay(x) for x in captured], dim=0)        # [rows, T, heads, gate_width]
        out.append(alpha.mean(dim=(0, 1)).reshape(-1).double().numpy())
    return out, len(layers)


def build():
    checkpoints = shortcut.local_checkpoints()
    models, parameters = [], {}
    for (init, variant, seed) in sorted(checkpoints):
        path = checkpoints[(init, variant, seed)]
        folder = path.parent.parent.name   # .../seed_<s>/<folder>/<variant>/latest_checkpoint.pt -> <folder>, a str
        rows = np.load(DIAG / f'overwrite_seed{seed}.npz')['tokens'][:ROWS]
        digest = hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest()
        tokens = torch.tensor(rows, dtype=torch.long)

        model = awg.load(path, variant, seed)
        total = sum(p.numel() for p in model.parameters())     # tied embedding counted once (as in the study doc)
        parameters.setdefault(variant, total)
        if parameters[variant] != total:
            raise RuntimeError(f'parameter count differs between seeds for {variant}')
        trained, layer_count = trained_alpha(model, tokens)

        calibration = None
        if init == 'mean-decay':
            calibration = json.loads((path.parent / 'calibration.json').read_text(encoding='utf-8'))
            if calibration.get('tokens_sha256') != digest:
                raise RuntimeError(f'calibration inputs do not match the saved training rows for seed {seed}')
            if calibration['tokens'] != rows.size:
                raise RuntimeError('calibration token count differs from the saved rows')

        layers = []
        for index in range(layer_count):
            t = trained[index]
            entry = dict(layer=index + 1,
                         trained=dict(mean=float(t.mean()), per_channel=_rounded(t),
                                      channel_spread=float(t.std()) if t.size > 1 else 0.0),
                         start=None)
            if calibration is not None:
                recorded = calibration['layers'][index]
                s = np.array(recorded['achieved_mean_decay'], dtype=float).reshape(-1)
                entry['start'] = dict(half_life=float(recorded['target_half_life']), mean=float(s.mean()),
                                      per_channel=_rounded(s), channel_spread=float(s.std()) if s.size > 1 else 0.0,
                                      maximum_error=float(recorded['maximum_error']))
            layers.append(entry)

        models.append(dict(init=init, variant=variant, seed=seed, folder=folder,
                           checkpoint=path.relative_to(ROOT).as_posix(), input_sha256=digest, layers=layers))
        print(f'{init:10s} {variant:12s} seed {seed}: {layer_count} layers, rows sha {digest[:12]}')

    return dict(
        generated=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        rows=ROWS,
        definition=('trained: mean alpha over the first 256 training rows of the seed, all tokens; '
                    'start: achieved means recorded in calibration.json (mean-decay only); '
                    'channel_spread: standard deviation across channels of the per-channel means (0 for scalar).'),
        parameters=parameters,
        models=models,
        sources=[
            'output/calibrated_recipe_study/matched-decay-v1/*/overwrite/seed_*/*/*/latest_checkpoint.pt',
            'output/calibrated_recipe_study/matched-decay-v1/*/overwrite/seed_*/calibrated/*/calibration.json',
            'research_results/three_seed_confirmation_20261117/diagnostic_training_samples/overwrite_seed*.npz',
        ],
    )


if __name__ == '__main__':
    result = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('wrote', OUT)
