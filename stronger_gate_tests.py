"""Fixed-weight gate shuffling, static-gate, and layer-local interventions."""
from contextlib import contextmanager
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import torch
from artifacts import atomic_write_json
from ablate_recipe_gates import probe_data
from confirm_recipe_study import select_runs
from diagnose_recipe_gates import parameter_fingerprint
from recipe_evaluation import evaluate
from recipe_model import RecipeLM,RecipeGDN


@torch.no_grad()
def static_means(model,data):
    values={};handles=[];modes={m:m.training for m in model.modules()}
    def capture(layer,inputs):
        values.setdefault(layer.layer_idx,[]).append(layer.decay(inputs[0]).detach())
    tokens=data.tokens.clone();tokens[data.erase_answers]=1
    try:
        model.eval()
        for layer in model.modules():
            if isinstance(layer,RecipeGDN):handles.append(layer.register_forward_pre_hook(capture))
        for start in range(0,len(tokens),32):model.network.backbone(tokens[start:start+32])
    finally:
        for h in handles:h.remove()
        for m,mode in modes.items():m.training=mode
    return {layer:torch.cat(parts).mean((0,1)) for layer,parts in values.items()}


@contextmanager
def intervention(model,mode,layer_index=None,seed=90221,means=None):
    saved=[]
    for layer in model.modules():
        if not isinstance(layer,RecipeGDN) or (layer_index is not None and layer.layer_idx!=layer_index):continue
        original=layer.decay;had_attribute='decay' in layer.__dict__;saved.append((layer,original,had_attribute))
        permutation=torch.randperm(layer.gate_width,generator=torch.Generator().manual_seed(seed+layer.layer_idx))
        constant=means[layer.layer_idx] if mode=='static' else None
        def changed(x,original=original,permutation=permutation,constant=constant):
            if mode=='static':return constant.to(x.device).view(1,1,*constant.shape).expand(x.shape[0],x.shape[1],*constant.shape)
            alpha=original(x)
            if mode=='shuffle':return alpha[...,permutation.to(x.device)]
            if mode=='mean':return alpha.mean(-1,keepdim=True).expand_as(alpha)
            raise ValueError('Unknown intervention')
        layer.decay=changed
    try:yield
    finally:
        for layer,original,had_attribute in saved:
            if had_attribute:layer.decay=original
            else:delattr(layer,'decay')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--confirmation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    selected=select_runs(a.root,['mqar','overwrite'],[42,43,44],[],'explore')
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(a.output/'protocol.json',dict(device='cpu',models=12,examples=256,
        fitting_training_indices=[0,256],evaluation_training_indices=[256,512],conditioning='answer_free',
        tests=['baseline','global mean','global static','global shuffle seeds90221/90222/90223',
               'mean one layer at a time layers0/1/2/3'],
        static_policy='per-channel means fitted on separate training-prefix inputs under original trained model, no labels/loss fitting',
        permutation_policy='one fixed permutation per layer for all tokens; preserve local gate multiset',
        interpretation='Post-hoc functional reliance, not trained architecture superiority or causal learning-speed explanation'))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    report=dict(status='running',models=[]);started=time.monotonic()
    for (family,seed,init,variant),(path,s) in sorted(selected.items()):
        if variant!='channel-wise':continue
        fit=probe_data(family,seed,a.confirmation,0,256);data=probe_data(family,seed,a.confirmation)
        model=RecipeLM(variant,8192,64,128,'chunked',seed)
        checkpoint=path.parent/'latest_checkpoint.pt'
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if saved['settings']!=s['settings']:raise ValueError('Checkpoint settings differ')
        model.load_state_dict(saved['model'],strict=True);del saved
        fingerprint=parameter_fingerprint(model);means=static_means(model,fit)
        baseline=evaluate(model,data,torch.device('cpu'))
        row=dict(family=family,seed=seed,initialization=init,variant=variant,baseline=baseline,
                 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 fit_data=fit.identity,probe_data=data.identity,tests=[])
        tests=[('global_mean','mean',None,0),('global_static','static',None,0)]+[
            (f'global_shuffle_{s}','shuffle',None,s) for s in [90221,90222,90223]]+[
            (f'layer_{layer}_mean','mean',layer,0) for layer in range(4)]
        for name,mode,layer,random_seed in tests:
            with intervention(model,mode,layer,random_seed,means):metrics=evaluate(model,data,torch.device('cpu'))
            if parameter_fingerprint(model)!=fingerprint:raise ValueError('Intervention changed weights')
            row['tests'].append(dict(test=name,metrics=metrics,
                                    accuracy_drop=baseline['exact_query_accuracy']-metrics['exact_query_accuracy']))
        report['models'].append(row);report['wall_seconds']=time.monotonic()-started
        atomic_write_json(a.output/'results.json',report)
        print(f'Interventions {len(report["models"])}/12: {family}/seed{seed}/{init}',flush=True)
    report.update(status='complete',wall_seconds=time.monotonic()-started);atomic_write_json(a.output/'results.json',report)
    rows=[]
    for model in report['models']:
        for result in model['tests']:
            rows.append({**{key:model[key] for key in ['family','seed','initialization']},'test':result['test'],
                'baseline_accuracy':model['baseline']['exact_query_accuracy'],
                'accuracy':result['metrics']['exact_query_accuracy'],'accuracy_drop':result['accuracy_drop'],
                'latest_accuracy':result['metrics']['latest_value_accuracy'],
                'unrelated_retention':result['metrics']['unrelated_association_retention']})
    with (a.output/'models.csv').open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
