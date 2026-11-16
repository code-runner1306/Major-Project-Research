"""Training-only matched mean-decay initialization for the published recipe."""
import copy
import hashlib
import math
import time

import torch
from torch.nn import functional as F
from recipe_model import RecipeGDN

TARGETS = (16., 32., 64., 128.)


def fit_offset(raw, rate, target, tolerance=1e-6):
    raw, rate = raw.double().cpu(), rate.double().cpu()
    if not torch.isfinite(raw).all() or not torch.isfinite(rate).all() or (rate <= 0).any():
        raise ValueError('Calibration requires finite activations and positive rates')
    if not 0 < target < 1:
        raise ValueError('Target mean decay must be strictly between zero and one')
    low, high = torch.full_like(rate,-80.), torch.full_like(rate,80.)
    def mean(offset):
        return torch.exp(-rate * F.softplus(raw + offset)).mean(0)
    if (mean(low) < target).any() or (mean(high) > target).any():
        raise ValueError('Calibration target is not bracketed')
    for _ in range(80):
        middle = (low+high)/2
        above = mean(middle) > target
        low, high = torch.where(above,middle,low), torch.where(above,high,middle)
        if (mean(middle)-target).abs().max() <= tolerance/10:
            break
    return middle.float()


@torch.no_grad()
def calibrate(model, tokens, targets=TARGETS, batch=32):
    layers = [m for m in model.modules() if isinstance(m,RecipeGDN)]
    if len(layers) != len(targets) or batch < 1 or not len(tokens):
        raise ValueError('One positive target per recurrent layer and nonempty training tokens required')
    if any(not math.isfinite(h) or h <= 0 for h in targets):
        raise ValueError('Invalid decay half-life target')
    start = time.monotonic()
    modes = {m:m.training for m in model.modules()}
    device = next(model.parameters()).device
    original = {name:p.detach().clone() for name,p in model.named_parameters() if name.endswith('dt_bias')}
    report = dict(examples=len(tokens), tokens=tokens.numel(), source='first training examples; labels unused',
                  tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                  interpretation='Mean-decay diagnostic targets, not measured full-state memory lifetimes', layers=[])
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
            model.eval()
            for index,(layer,half_life) in enumerate(zip(layers,targets)):
                captured=[]
                def capture(module, inputs):
                    x=inputs[0]
                    captured.append(module.a_proj(x).reshape(-1,module.heads,module.gate_width).float().cpu())
                handle=layer.register_forward_pre_hook(capture)
                try:
                    for offset in range(0,len(tokens),batch):
                        model.network.backbone(tokens[offset:offset+batch].to(device))
                finally:
                    handle.remove()
                raw=torch.cat(captured)
                rate=layer.A_log.detach().float().exp().cpu()
                target=2**(-1/half_life)
                fitted=fit_offset(raw,rate,target)
                layer.dt_bias.copy_(fitted.to(device))
                alpha=torch.exp(-rate.double()*F.softplus(raw.double()+layer.dt_bias.detach().double().cpu()))
                achieved=alpha.mean(0)
                error=(achieved-target).abs().max().item()
                if not torch.isfinite(alpha).all() or error>1e-6:
                    raise RuntimeError('Calibration failed mean-decay tolerance')
                report['layers'].append(dict(layer=index,target_half_life=half_life,target_mean_decay=target,
                    achieved_mean_decay=achieved.tolist(),maximum_error=error,
                    decay_std=alpha.std(0,unbiased=False).tolist(),offsets=fitted.tolist()))
    except BaseException:
        for name,p in model.named_parameters():
            if name in original:
                p.copy_(original[name])
        raise
    finally:
        for module,training in modes.items():
            module.training=training
    report.update(passed=True,wall_seconds=time.monotonic()-start)
    return report


def qualify_calibrated(model,tokens,steps=5):
    """Reference/full projection versus chunked/scored loss after calibration."""
    reference=copy.deepcopy(model)
    candidate=copy.deepcopy(model)
    for layer in reference.modules():
        if isinstance(layer,RecipeGDN): layer.backend='reference'
    for layer in candidate.modules():
        if isinstance(layer,RecipeGDN): layer.backend='chunked'
    device=next(model.parameters()).device
    optimizers=[m.optimizer(1e-3) for m in (reference,candidate)]
    maximum=relative=0.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
        for step in range(steps):
            labels=torch.full_like(tokens,-100)
            labels[:,[15,23]]=tokens[:,[16,24]]
            for m,opt in zip((reference,candidate),optimizers):
                m.train();torch.manual_seed(12000+step);opt.zero_grad(set_to_none=True)
                if m is reference: _,loss=m(tokens.to(device),labels.to(device))
                else: loss=m.loss(tokens.to(device),labels.to(device))
                loss.backward()
                if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):
                    raise RuntimeError('Nonfinite calibrated qualification trajectory')
                opt.step()
            for actual,expected in zip(candidate.parameters(),reference.parameters()):
                maximum=max(maximum,(actual-expected).abs().max().item())
                relative=max(relative,((actual-expected).norm()/expected.norm().clamp_min(1e-12)).item())
                if maximum>1e-4 or relative>5e-4:
                    raise RuntimeError('Calibrated reference/chunked optimizer drift exceeds tolerance')
    return dict(passed=True,steps=steps,max_parameter_absolute_error=maximum,max_parameter_relative_error=relative)
