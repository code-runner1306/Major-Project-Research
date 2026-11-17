"""Predeclared memory-load/context/distance/rewrite stress tests, no training."""
import argparse
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import time

import torch
from artifacts import atomic_write_json
from config import RetrievalConfig
from confirm_recipe_study import select_runs
from recipe_data import RecipeData,mqar,manifest,audit_mqar,build_data,split_overlap
from recipe_evaluation import evaluate
from recipe_model import RecipeLM
from retrieval import RetrievalDataset,evaluate_retrieval_baselines

MQAR_CELLS=[('baseline',64,4),('more_keys',64,8),('longer_context',128,4),('longer_more_keys',128,8)]
OVERWRITE_CELLS=[('baseline',4,8,1),('more_keys',8,8,1),('farther_query',4,24,1),('more_rewrites',4,8,3)]


@dataclass
class StressCase:
    data:RecipeData
    stale_values:torch.Tensor
    audit:dict


def make_mqar(examples,seed,length,keys):
    tokens,labels=mqar(examples,8192,length,keys,seed)
    data=RecipeData(tokens,labels,torch.where(labels!=-100,0,-1),torch.full_like(labels,-1),
                    torch.zeros_like(tokens,dtype=torch.bool),manifest(tokens,labels,seed=seed,split='stress',keys=keys))
    audit=audit_mqar(tokens,labels,keys)
    if not audit['passed']:raise ValueError('MQAR stress audit failed')
    return StressCase(data,torch.full((*labels.shape,1),-1),audit)


def make_overwrite(examples,seed,keys,distance,rewrites,train_seed):
    cfg=RetrievalConfig(vocabulary_size=8192,sequence_lengths=(65,),key_counts=(keys,),query_distances=(distance,),
                        overwrite_counts=(rewrites,),generator_version='mqar-overwrite-v4')
    dataset=RetrievalDataset(task='overwrite',examples=examples,config=cfg,seed=seed,split='held_out')
    records=[dataset.example(i) for i in range(examples)]
    tokens=torch.stack([x.tokens for x in records]);labels=torch.stack([x.labels for x in records])
    kinds=torch.full_like(labels,-1);stale=torch.full_like(labels,-1)
    all_stale=torch.full((*labels.shape,rewrites),-1);erase=torch.zeros_like(tokens,dtype=torch.bool)
    for index,record in enumerate(records):
        for answer in record.answers:
            pos=answer.label_position;kinds[index,pos]=1 if answer.kind=='latest' else 2
            if answer.kind=='latest':
                stale[index,pos]=answer.stale_values[-1]
                all_stale[index,pos,:len(answer.stale_values)]=torch.tensor(answer.stale_values)
            if pos+1<tokens.shape[1]:erase[index,pos+1]=True
    frequency_cfg=RetrievalConfig(vocabulary_size=8192,sequence_lengths=(65,),key_counts=(4,),query_distances=(8,),
                                  overwrite_counts=(1,),generator_version='mqar-overwrite-v4')
    frequency=RetrievalDataset(task='overwrite',examples=256,config=frequency_cfg,seed=train_seed,split='train')
    audit=evaluate_retrieval_baselines(dataset,frequency_dataset=frequency)
    audit['passed']=(audit['status']=='complete' and audit['oracle_audit']['pass'] and
                    audit['counterfactual_query_key_audit']['pass'] and audit['shortcut_criterion']['all_cells_pass'])
    audit['post_target_write_count_mean']=sum(r.metadata['post_target_write_count'] for r in records)/examples
    if not audit['passed']:raise ValueError('Overwrite stress audit failed')
    identity=manifest(tokens,labels,seed=seed,split='stress',keys=keys,distance=distance,rewrites=rewrites,
                      source_manifest=dataset.manifest())
    return StressCase(RecipeData(tokens,labels,kinds,stale,erase,identity),all_stale,audit)


def stale_rate(predictions,kinds,all_stale):
    latest=kinds==1
    hits=((predictions[:,None]==all_stale)&(all_stale>=0)).any(-1)&latest
    return hits.sum().item()/latest.sum().item() if latest.any() else None


def evaluate_case(model,case):
    predictions=[];original=model.scored_logits;had_attribute='scored_logits' in model.__dict__
    def capture(tokens,mask):
        logits=original(tokens,mask);predictions.append(logits.argmax(-1).detach().cpu());return logits
    model.scored_logits=capture
    try:result=evaluate(model,case.data,torch.device('cpu'))
    finally:
        if had_attribute:model.scored_logits=original
        else:delattr(model,'scored_logits')
    mask=case.data.labels!=-100
    result['stale_value_error_rate']=stale_rate(torch.cat(predictions),case.data.kinds[mask],case.stale_values[mask])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,default=256);a=p.parse_args()
    if a.examples<1:p.error('Examples must be positive')
    selected=select_runs(a.root,['mqar','overwrite'],[42,43,44],[],'explore')
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(a.output/'protocol.json',dict(device='cpu',examples=a.examples,seed_offset=500000,
        mqar_cells=MQAR_CELLS,overwrite_cells=OVERWRITE_CELLS,conditioning='answer_free',
        stale_definition='prediction equals ANY prior value for the updated key',
        data_policy='same examples per task/seed/cell across conditions; generated diagnostic data, not tuning data',
        interpretation='Zero-shot distribution-shift stress tests of fixed checkpoints, not learnability of newly trained hard-task models'))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    report=dict(status='running',results=[],data_audits=[]);started=time.monotonic()
    for family in ['mqar','overwrite']:
        for seed in [42,43,44]:
            cells=MQAR_CELLS if family=='mqar' else OVERWRITE_CELLS
            cases={}
            for cell in cells:
                case=(make_mqar(a.examples,seed+500000,cell[1],cell[2]) if family=='mqar' else
                      make_overwrite(a.examples,seed+500000,cell[1],cell[2],cell[3],seed+100000))
                cases[cell[0]]=case
            # Only baseline can equal an original training example in shape and answer count.
            train=build_data(family,100000,seed+100000,'train',8192,64)
            for name,case in cases.items():
                if name=='baseline':overlap=split_overlap(train,case.data)
                elif family=='mqar':overlap=0  # different sequence length or number of supervised answers
                else:overlap=split_overlap(train,case.data)
                if overlap:raise ValueError('Stress cases overlap training data')
                report['data_audits'].append(dict(family=family,seed=seed,cell=name,identity=case.data.identity,
                                                train_overlap=overlap,audit=case.audit))
            del train
            atomic_write_json(a.output/'results.json',report)
            for init in ['default','calibrated']:
                for variant in ['scalar','channel-wise']:
                    path,s=selected[family,seed,init,variant]
                    model=RecipeLM(variant,8192,64,128,'chunked',seed)
                    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                        saved=torch.load(path.parent/'latest_checkpoint.pt',map_location='cpu',weights_only=True)
                    if saved['settings']!=s['settings']:raise ValueError('Checkpoint settings differ')
                    model.load_state_dict(saved['model'],strict=True);del saved
                    for cell,case in cases.items():
                        metrics=evaluate_case(model,case)
                        report['results'].append(dict(family=family,seed=seed,initialization=init,variant=variant,cell=cell,metrics=metrics))
                    report['wall_seconds']=time.monotonic()-started;atomic_write_json(a.output/'results.json',report)
                    print(f'Stress models {len(report["results"])//4}/24: {family}/seed{seed}/{init}/{variant}',flush=True)
                    del model
    report.update(status='complete',wall_seconds=time.monotonic()-started);atomic_write_json(a.output/'results.json',report)
    rows=[]
    for r in report['results']:
        rows.append({**{k:r[k] for k in ['family','seed','initialization','variant','cell']},
                     **{k:r['metrics'][k] for k in ['exact_query_accuracy','latest_value_accuracy','unrelated_association_retention','stale_value_error_rate']}})
    with (a.output/'models.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
