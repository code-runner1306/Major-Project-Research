"""Evaluate a complete selected study matrix once, after collecting every run."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from artifacts import atomic_write_json
from recipe_data import build_data,split_overlap,audit_mqar
from recipe_evaluation import evaluate
from recipe_model import RecipeLM
from run_recipe_study import identity,PROTOCOL


def select_runs(root,families,seeds,run_ids):
    selected={}
    for path in root.rglob('summary.json'):
        s=json.loads(path.read_text());cfg=s.get('settings',{})
        if cfg.get('protocol')!=PROTOCOL or cfg.get('stage')!='study':continue
        if cfg['family'] not in families or cfg['seed'] not in seeds:continue
        if run_ids and not any(part in run_ids for part in path.parts):continue
        key=(cfg['family'],cfg['seed'],cfg['initialization'],cfg['variant'])
        if key in selected:raise ValueError('Duplicate study condition; select the intended --run-id values')
        if s['status']!='complete' or cfg['epochs']!=64 or s['accepted_optimizer_steps']!=12544:
            raise ValueError('Only complete 64-epoch study runs qualify')
        if cfg['source_hashes']!=identity():raise ValueError('Study code differs from checkpoint source')
        selected[key]=(path,s)
    expected={(f,s,i,v) for f in families for s in seeds for i in ['default','calibrated'] for v in ['scalar','channel-wise']}
    if set(selected)!=expected:raise ValueError(f'Incomplete study matrix; missing: {sorted(expected-set(selected))}')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',nargs='+',choices=['mqar','overwrite'],default=['mqar','overwrite'])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--run-id',nargs='*',default=[])
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    a=p.parse_args()
    selected=select_runs(a.root,a.family,a.seeds,a.run_id)
    a.output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(a.output/'confirmation_protocol.json',dict(protocol=PROTOCOL,families=a.family,seeds=a.seeds,
        examples=512,seed_offset=400000,overwrite_domain='held_out',selection='all final checkpoints; no best selection'))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    report=dict(status='running',results=[],contrasts=[],device=str(device));started=time.monotonic()
    try:
        for family in a.family:
            for seed in a.seeds:
                rows=[];paired=None;shared=None;base_parameters={}
                cfg=selected[(family,seed,'default','scalar')][1]['settings']
                split='held_out' if family=='overwrite' else 'confirmation'
                data=build_data(family,512,seed+400000,split,cfg['vocabulary'],cfg['length'])
                overlaps={}
                for name,count,offset,domain in [('train',100000,100000,'train'),('development',512,200000,'validation'),
                    ('previous_confirmation',512,300000,split)]:
                    old=build_data(family,count,seed+offset,domain,cfg['vocabulary'],cfg['length'])
                    overlaps[name]=split_overlap(old,data);del old
                if any(overlaps.values()):raise ValueError('Confirmation overlaps existing data')
                if family=='mqar' and not audit_mqar(data.tokens,data.labels)['passed']:raise ValueError('MQAR audit failed')
                atomic_write_json(a.output/f'{family}_seed{seed}_data.json',dict(identity=data.identity,overlap=overlaps))
                for initialization in ['default','calibrated']:
                    for variant in ['scalar','channel-wise']:
                        path,summary=selected[(family,seed,initialization,variant)]
                        manifests=json.loads((path.parent/'data_manifests.json').read_text())
                        if paired is not None and manifests!=paired:raise ValueError('Condition datasets differ')
                        paired=manifests
                        initial=json.loads((path.parent/'initialization_manifest.json').read_text())
                        if shared is not None and shared!=initial['shared_parameter_sha256']:
                            raise ValueError('Shared backbone initialization differs across conditions')
                        shared=initial['shared_parameter_sha256']
                        if variant in base_parameters and base_parameters[variant]!=initial['base_parameter_sha256']:
                            raise ValueError('Default/calibrated base model differs')
                        base_parameters[variant]=initial['base_parameter_sha256']
                        checkpoint=path.parent/'latest_checkpoint.pt'
                        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                        if saved['settings']!=summary['settings'] or saved['completed_steps']!=12544 or saved['completed_epochs']!=64:
                            raise ValueError('Checkpoint and summary differ')
                        model=RecipeLM(variant,cfg['vocabulary'],cfg['length'],cfg['width'],'chunked',seed)
                        model.load_state_dict(saved['model'],strict=True);del saved
                        model.to(device)
                        primary=evaluate(model,data,device)
                        secondary=evaluate(model,data,device,history='teacher_forced') if family=='overwrite' else dict(primary,history_conditioning='teacher_forced')
                        row=dict(family=family,seed=seed,initialization=initialization,variant=variant,answer_free=primary,
                                 teacher_forced=secondary,learning=summary['learning'],checkpoint=str(checkpoint),
                                 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                        rows.append(row);report['results'].append(row)
                        atomic_write_json(a.output/'confirmation_results.json',report)
                        del model
                        if device.type=='cuda':torch.cuda.empty_cache()
                key='accuracy_aulc_epochs1_16' if family=='mqar' else 'retention_aulc_epochs1_16'
                scores={(row['initialization'],row['variant']):row['learning'][key] for row in rows}
                default=scores['default','channel-wise']-scores['default','scalar']
                calibrated=scores['calibrated','channel-wise']-scores['calibrated','scalar']
                report['contrasts'].append(dict(family=family,seed=seed,metric=key,channel_minus_scalar_default=default,
                    channel_minus_scalar_calibrated=calibrated,interaction=calibrated-default))
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        atomic_write_json(a.output/'confirmation_results.json',report)


if __name__=='__main__':main()
