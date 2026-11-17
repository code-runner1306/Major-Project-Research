"""Post-hoc shared-mean gate intervention on fixed training-side probe examples."""
from contextlib import contextmanager
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from artifacts import atomic_write_json
from confirm_recipe_study import select_runs
from diagnose_recipe_gates import parameter_fingerprint
from recipe_data import RecipeData,build_data,manifest
from recipe_evaluation import evaluate
from recipe_model import RecipeLM,RecipeGDN


@contextmanager
def shared_mean_gates(model):
    saved=[]
    for layer in model.modules():
        if isinstance(layer,RecipeGDN):
            prior=layer.decay
            had_attribute='decay' in layer.__dict__
            saved.append((layer,prior,had_attribute))
            def shared(x,original=prior):
                alpha=original(x)
                return alpha.mean(-1,keepdim=True).expand_as(alpha)
            layer.decay=shared
    try:yield
    finally:
        for layer,prior,had_attribute in saved:
            if had_attribute:layer.decay=prior
            else:delattr(layer,'decay')


def probe_data(family,seed,confirmation):
    with np.load(confirmation/'diagnostic_training_samples'/f'{family}_seed{seed}.npz',allow_pickle=False) as sample:
        tokens=torch.from_numpy(sample['tokens'].copy());labels=torch.from_numpy(sample['labels'].copy())
        erase=torch.from_numpy(sample['erase_answers'].copy())
    if family=='overwrite':
        prefix=build_data(family,512,seed+100000,'train',8192,64)
        if not torch.equal(prefix.tokens,tokens) or not torch.equal(prefix.labels,labels) or not torch.equal(prefix.erase_answers,erase):
            raise ValueError('Overwrite diagnostic prefix does not match cached full training pool')
        kinds=prefix.kinds[256:];stale=prefix.stale[256:]
    else:
        kinds=torch.where(labels[256:]!=-100,0,-1);stale=torch.full_like(labels[256:],-1)
    return RecipeData(tokens[256:],labels[256:],kinds,stale,erase[256:],
                      manifest(tokens[256:],labels[256:],seed=seed+100000,split='training_diagnostic',indices=[256,512]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--confirmation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if json.loads((a.confirmation/'confirmation_results.json').read_text())['status']!='complete':
        raise ValueError('Confirmation is incomplete')
    selected=select_runs(a.root,['mqar','overwrite'],[42,43,44],[],'explore')
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(a.output/'protocol.json',dict(device='cpu',examples=256,training_indices=[256,512],
        intervention='replace alpha at every layer by its per-token headwise channel mean, preserving instantaneous mean',
        weights='fixed',initialization='unchanged trained checkpoint',conditioning='answer_free',
        interpretation='Post-hoc reliance test on training inputs, not a fair trained-scalar comparison or a causal explanation of learning speed'))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    results=[];started=time.monotonic()
    for (family,seed,initialization,variant),(path,summary) in sorted(selected.items()):
        if variant!='channel-wise':continue
        cfg=summary['settings'];data=probe_data(family,seed,a.confirmation)
        model=RecipeLM(variant,8192,64,128,'chunked',seed)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint=torch.load(path.parent/'latest_checkpoint.pt',map_location='cpu',weights_only=True)
        if checkpoint['settings']!=cfg:raise ValueError('Checkpoint settings differ')
        model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
        before=parameter_fingerprint(model)
        baseline=evaluate(model,data,torch.device('cpu'))
        with shared_mean_gates(model):
            intervened=evaluate(model,data,torch.device('cpu'))
        if parameter_fingerprint(model)!=before:raise ValueError('Intervention changed learned parameters')
        results.append(dict(family=family,seed=seed,initialization=initialization,variant=variant,
            baseline=baseline,shared_mean=intervened,accuracy_drop=baseline['exact_query_accuracy']-intervened['exact_query_accuracy']))
        atomic_write_json(a.output/'results.json',dict(status='running',results=results,wall_seconds=time.monotonic()-started))
        print(f'{family}/seed{seed}/{initialization}: original {baseline["exact_query_accuracy"]:.3%}, '
              f'shared mean {intervened["exact_query_accuracy"]:.3%}',flush=True)
    atomic_write_json(a.output/'results.json',dict(status='complete',results=results,wall_seconds=time.monotonic()-started))
    flat=[dict(family=r['family'],seed=r['seed'],initialization=r['initialization'],
        baseline_accuracy=r['baseline']['exact_query_accuracy'],shared_mean_accuracy=r['shared_mean']['exact_query_accuracy'],
        accuracy_drop=r['accuracy_drop'],baseline_retention=r['baseline']['unrelated_association_retention'],
        shared_mean_retention=r['shared_mean']['unrelated_association_retention']) for r in results]
    with (a.output/'models.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=flat[0].keys());writer.writeheader();writer.writerows(flat)


if __name__=='__main__':main()
