"""DAMI-style joint initialization adapted to the existing one-head study.

Equations 10–11: https://arxiv.org/html/2610.41994v2
Our extension matches each channel independently, keeps the previous layer
half-life targets, and uses a four-layer .25/.75/.75/.25 sensitivity profile.
This is not an exact reproduction of the paper's multihead spectrum.
"""
import hashlib
import math
import time

import torch
from torch.nn import functional as F
from recipe_model import RecipeGDN
from recipe_calibration import TARGETS

SENSITIVITIES = (.25, .75, .75, .25)


def fit_joint(raw, half_life, sensitivity):
    raw = raw.detach().double().cpu()
    if not torch.isfinite(raw).all() or not raw.numel():
        raise ValueError('Finite nonempty projected activations required')
    if not math.isfinite(half_life) or half_life <= 0 or not 0 < sensitivity < 1:
        raise ValueError('Invalid joint targets')
    low, high = -raw.amax(0)-80, -raw.amin(0)+80
    for _ in range(80):
        middle = (low+high)/2
        below = (raw+middle).sigmoid().mean(0) < sensitivity
        low, high = torch.where(below,middle,low), torch.where(below,high,middle)
    bias = ((low+high)/2).float()
    # E[log(alpha)] = -log(2)/half_life, not E[alpha] matching.
    rate = (math.log(2)/half_life)/F.softplus(raw+bias.double()).mean(0)
    return bias, rate.log().float()


@torch.no_grad()
def calibrate_joint(model,tokens,targets=TARGETS,batch=32):
    layers = [m for m in model.modules() if isinstance(m,RecipeGDN)]
    if len(layers)!=4 or len(targets)!=4 or not len(tokens) or batch<1:
        raise ValueError('Joint protocol requires four layers and training examples')
    start=time.monotonic();device=next(model.parameters()).device
    modes={m:m.training for m in model.modules()}
    original={name:p.detach().clone() for name,p in model.named_parameters()
              if name.endswith(('A_log','dt_bias'))}
    report=dict(method='joint-timescale-sensitivity-v1',source='first training examples; labels unused',
        examples=len(tokens),tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
        interpretation='Mean log-decay half-life and Softplus sensitivity; not full-state lifetime',layers=[])
    try:
        model.eval()
        for index,(layer,half_life,sensitivity) in enumerate(zip(layers,targets,SENSITIVITIES)):
            captured=[]
            def capture(module,inputs):
                captured.append(module.a_proj(inputs[0]).reshape(-1,module.heads,module.gate_width).cpu())
            handle=layer.register_forward_pre_hook(capture)
            try:
                for offset in range(0,len(tokens),batch):
                    model.network.backbone(tokens[offset:offset+batch].to(device))
            finally:
                handle.remove()
            raw=torch.cat(captured).double()
            bias,log_rate=fit_joint(raw,half_life,sensitivity)
            layer.dt_bias.copy_(bias.to(device));layer.A_log.copy_(log_rate.to(device))
            softplus=F.softplus(raw+bias.double())
            achieved_s=(raw+bias.double()).sigmoid().mean(0)
            achieved_log_decay=-log_rate.double().exp()*softplus.mean(0)
            s_error=(achieved_s-sensitivity).abs().max().item()
            d_error=(achieved_log_decay+math.log(2)/half_life).abs().max().item()
            if not torch.isfinite(softplus).all() or max(s_error,d_error)>1e-6:
                raise RuntimeError('Joint calibration target tolerance failed')
            report['layers'].append(dict(layer=index,target_half_life=half_life,
                target_sensitivity=sensitivity,achieved_sensitivity=achieved_s.tolist(),
                achieved_mean_log_decay=achieved_log_decay.tolist(),sensitivity_error=s_error,
                log_decay_error=d_error,offsets=bias.tolist(),A_log=log_rate.tolist()))
    except BaseException:
        for name,p in model.named_parameters():
            if name in original:p.copy_(original[name])
        raise
    finally:
        for module,training in modes.items():module.training=training
    return dict(report,passed=True,wall_seconds=time.monotonic()-start)
