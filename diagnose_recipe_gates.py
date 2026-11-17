"""Training-side, inference-only diagnostics of initial and final study gates."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from artifacts import atomic_write_json
from confirm_recipe_study import select_runs
from recipe_model import RecipeLM,RecipeGDN


def parameter_fingerprint(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.named_parameters()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def describe(value):
    x=value.detach().double().reshape(-1)
    if not len(x):return None
    if not torch.isfinite(x).all():raise ValueError('Nonfinite diagnostic values')
    sample=x if len(x)<=16384 else x[torch.randint(len(x),(16384,),generator=torch.Generator().manual_seed(90211))]
    q=torch.quantile(sample,torch.tensor([.01,.5,.99],dtype=torch.float64))
    return dict(mean=x.mean().item(),std=x.std(unbiased=False).item(),minimum=x.min().item(),
                p01=q[0].item(),median=q[1].item(),p99=q[2].item(),maximum=x.max().item())


def sensitivity(raw,rate,offset):
    alpha=torch.exp(-rate*F.softplus(raw+offset))
    return alpha*rate*torch.sigmoid(raw+offset)


def gate_summary(alpha,beta,derivative,query_mask,filler_mask):
    a=alpha.double()
    gate_means=a.mean((0,1))
    flat=a.reshape(-1,a.shape[-1])  # one head in this protocol
    # Deterministic subsample; diversity is a gate-variation statistic, not capacity.
    index=torch.randperm(len(flat),generator=torch.Generator().manual_seed(90210))[:min(1024,len(flat))]
    centered=flat[index]-flat[index].mean(0)
    covariance=centered.T@centered/max(1,len(index))
    trace=covariance.diag().sum()
    rank=(trace.square()/covariance.square().sum().clamp_min(1e-30)).item()
    std=covariance.diag().sqrt()
    active=std>1e-8
    correlation=None
    if int(active.sum())>1:
        cov=covariance[active][:,active];scales=std[active]
        corr=cov/(scales[:,None]*scales[None,:])
        mask=~torch.eye(len(corr),dtype=torch.bool)
        correlation=corr[mask].abs().mean().item()
    mean=a.mean().item()
    h=math.log(.5)/math.log(mean) if 0<mean<1 else None
    result=dict(alpha=describe(a),beta=describe(beta),absolute_offset_sensitivity=describe(derivative),
        mean_derived_half_life=h,per_gate_mean_min=gate_means.min().item(),per_gate_mean_max=gate_means.max().item(),
        within_token_channel_std=a.std(-1,unbiased=False).mean().item(),
        mean_within_gate_token_std=flat.std(0,unbiased=False).mean().item(),
        gate_variation_effective_rank=rank,mean_absolute_active_gate_correlation=correlation,
        fraction_alpha_below_001=(a<=.01).double().mean().item(),
        fraction_alpha_above_0999=(a>=.999).double().mean().item(),
        fraction_beta_below_001=(beta<=.01).double().mean().item(),
        fraction_beta_above_099=(beta>=.99).double().mean().item())
    result['positions']={}
    for name,mask in [('query_key',query_mask),('filler',filler_mask),('other',~(query_mask|filler_mask))]:
        result['positions'][name]=dict(tokens=int(mask.sum()),alpha=describe(a[mask]),
                                      beta=describe(beta[mask]),absolute_offset_sensitivity=describe(derivative[mask]))
    return result


@torch.no_grad()
def probe(model,tokens,labels,family,batch=32):
    captured={};handles=[]
    def capture(module,inputs):
        x=inputs[0]
        raw=module.a_proj(x).view(x.size(0),x.size(1),module.heads,module.gate_width).float()
        a=module.decay(x)
        b=module.b_proj(x).view(x.size(0),x.size(1),module.heads,1).sigmoid()
        d=sensitivity(raw,module.A_log.float().exp(),module.dt_bias.float())
        captured.setdefault(module.layer_idx,[]).append((a.cpu(),b.cpu(),d.cpu()))
    modes={m:m.training for m in model.modules()}
    try:
        model.eval()
        for layer in model.modules():
            if isinstance(layer,RecipeGDN):handles.append(layer.register_forward_pre_hook(capture))
        for start in range(0,len(tokens),batch):
            model.network.backbone(tokens[start:start+batch])
    finally:
        for handle in handles:handle.remove()
        for module,mode in modes.items():module.training=mode
    query=labels!=-100
    filler=tokens==(0 if family=='mqar' else 1)
    return {layer:gate_summary(*(torch.cat([chunk[i] for chunk in values]) for i in range(3)),query,filler)
            for layer,values in captured.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--confirmation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    confirmation=json.loads((a.confirmation/'confirmation_results.json').read_text())
    if confirmation['status']!='complete' or len(confirmation['results'])!=24:
        raise ValueError('Finish the complete confirmation evaluation before gate diagnostics')
    selected=select_runs(a.root,['mqar','overwrite'],[42,43,44],[],'explore')
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=dict(models=24,states=['final','initial only if base fingerprint verifies'],calibration_positions=[0,256],independent_positions=[256,512],
        corpora=['calibration_training','independent_training','independent_answer_free'],
        source='first512 examples of exact regenerated full100k training pool; no confirmation examples used',
        conditioning='original inputs for training corpora; previous gold answers erased for answer_free',
        batch=32,dropout=False,gradients=False,device='cpu',correlation_sample_tokens=1024,correlation_sample_seed=90210,
        quantile_sample_max=16384,quantile_sample_seed=90211,
        statistics='means,standard deviations,extrema,fractions exact; large-tensor quantiles use fixed subsample',
        sensitivity='absolute d_alpha/d_raw = absolute d_alpha/d_dt_bias, not a full loss gradient',
        initial_replay_policy='On fingerprint mismatch, do not probe a reconstructed initial model. Use saved calibration means/std/offsets only.',
        interpretation='Descriptive gate behavior; decay half-lives and diversity are not measured memory lifetimes or causal explanations')
    atomic_write_json(a.output/'diagnostics_protocol.json',protocol)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    rows=[];drifts=[];sources=[];recorded_initial=[];started=time.monotonic()
    for (family,seed,initialization,variant),(path,summary) in sorted(selected.items()):
        cfg=summary['settings']
        sample=a.confirmation/'diagnostic_training_samples'/f'{family}_seed{seed}.npz'
        with np.load(sample,allow_pickle=False) as data:
            tokens=torch.from_numpy(data['tokens'].copy());labels=torch.from_numpy(data['labels'].copy())
            erase=torch.from_numpy(data['erase_answers'].copy())
        if tokens.shape!=(512,64):raise ValueError('Unexpected diagnostic sample shape')
        model=RecipeLM(variant,cfg['vocabulary'],cfg['length'],cfg['width'],'chunked',seed)
        expected=json.loads((path.parent/'initialization_manifest.json').read_text())
        actual_base=parameter_fingerprint(model)
        replay_verified=actual_base==expected['base_parameter_sha256']
        calibration=None
        if initialization=='calibrated':
            calibration=json.loads((path.parent/'calibration.json').read_text())
            token_hash=hashlib.sha256(tokens[:256].contiguous().numpy().tobytes()).hexdigest()
            if token_hash!=calibration['tokens_sha256']:raise ValueError('Calibration input hash differs')
            with torch.no_grad():
                for layer in model.modules():
                    if isinstance(layer,RecipeGDN):
                        entry=calibration['layers'][layer.layer_idx]
                        layer.dt_bias.copy_(torch.tensor(entry['offsets'],dtype=layer.dt_bias.dtype))
            recorded_initial.append(dict(family=family,seed=seed,initialization=initialization,variant=variant,
                                         source=str(path.parent/'calibration.json'),layers=calibration['layers']))
        initial_parameters={name:param.detach().clone() for name,param in model.named_parameters()
                            if (replay_verified and name.endswith(('dt_bias','A_log','a_proj.weight','b_proj.weight')))
                            or (calibration is not None and name.endswith('dt_bias'))}
        checkpoint=path.parent/'latest_checkpoint.pt'
        checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        sources.append(dict(family=family,seed=seed,initialization=initialization,variant=variant,
                            checkpoint=str(checkpoint),sha256=checkpoint_hash,sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
                            initial_replay_verified=replay_verified,saved_base_sha256=expected['base_parameter_sha256'],
                            reconstructed_base_sha256=actual_base))
        independent=tokens[256:].clone();independent[erase[256:]]=1
        corpora=[('calibration_training',tokens[:256],labels[:256]),
                 ('independent_training',tokens[256:],labels[256:])]
        if family=='overwrite':corpora.append(('independent_answer_free',independent,labels[256:]))
        for state in (['initial','final'] if replay_verified else ['final']):
            if state=='final':
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if saved['settings']!=cfg:raise ValueError('Checkpoint settings differ')
                model.load_state_dict(saved['model'],strict=True);del saved
                for name,param in model.named_parameters():
                    if name in initial_parameters:
                        initial=initial_parameters[name]
                        delta=param.detach()-initial
                        drifts.append(dict(family=family,seed=seed,initialization=initialization,variant=variant,
                            parameter=name,initial_mean=initial.mean().item(),final_mean=param.detach().mean().item(),
                            mean_delta=delta.mean().item(),rms_delta=delta.square().mean().sqrt().item(),
                            initial_source='verified replay' if replay_verified else 'saved calibration offset'))
            for corpus,input_tokens,target_labels in corpora:
                measured=probe(model,input_tokens,target_labels,family)
                for layer,values in measured.items():
                    if state=='initial' and calibration and corpus=='calibration_training':
                        entry=calibration['layers'][layer]
                        if max(abs(values['per_gate_mean_min']-entry['target_mean_decay']),
                               abs(values['per_gate_mean_max']-entry['target_mean_decay']))>1e-6:
                            raise ValueError('Replayed calibrated initial means do not match targets')
                    rows.append(dict(family=family,seed=seed,initialization=initialization,variant=variant,
                                     state=state,corpus=corpus,layer=layer,**values))
        atomic_write_json(a.output/'gate_diagnostics.json',dict(status='running',models_completed=len(sources),
            sources=sources,measurements=rows,recorded_initial_calibration=recorded_initial,
            parameter_drift=drifts,wall_seconds=time.monotonic()-started))
        print(f'Gate diagnostics {len(sources)}/24: {family}/seed{seed}/{initialization}/{variant}',flush=True)
        del model
    atomic_write_json(a.output/'gate_diagnostics.json',dict(status='complete',models_completed=len(sources),
        sources=sources,measurements=rows,recorded_initial_calibration=recorded_initial,
        parameter_drift=drifts,wall_seconds=time.monotonic()-started))
    flat=[]
    for row in rows:
        flat.append({**{key:row[key] for key in ['family','seed','initialization','variant','state','corpus','layer']},
            **{key:row[key] for key in ['mean_derived_half_life','within_token_channel_std','mean_within_gate_token_std',
                'gate_variation_effective_rank','mean_absolute_active_gate_correlation','fraction_alpha_above_0999']},
            **{prefix+'_'+key:value for prefix in ['alpha','beta','absolute_offset_sensitivity'] for key,value in row[prefix].items()}})
    for name,items in [('gates.csv',flat),('parameter_drift.csv',drifts)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=items[0].keys());writer.writeheader();writer.writerows(items)


if __name__=='__main__':main()
