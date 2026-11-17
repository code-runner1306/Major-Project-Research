"""Evaluate a complete selected study matrix once, after collecting every run."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import torch
from artifacts import atomic_write_json
from recipe_data import build_data,split_overlap,audit_mqar
from recipe_evaluation import evaluate
from recipe_model import RecipeLM
from run_recipe_study import identity,PROTOCOL


def same_manifest(left,right):
    """Compare JSON-persisted metadata without confusing tuple/list encoding."""
    return json.loads(json.dumps(left))==json.loads(json.dumps(right))


def select_runs(root,families,seeds,run_ids,stage='study'):
    expected_epochs={'study':64,'explore':16}[stage]
    expected_steps=196*expected_epochs
    selected={}
    for path in root.rglob('summary.json'):
        s=json.loads(path.read_text());cfg=s.get('settings',{})
        if cfg.get('protocol')!=PROTOCOL or cfg.get('stage')!=stage:continue
        if cfg['family'] not in families or cfg['seed'] not in seeds:continue
        if run_ids and not any(part in run_ids for part in path.parts):continue
        key=(cfg['family'],cfg['seed'],cfg['initialization'],cfg['variant'])
        if key in selected:raise ValueError('Duplicate study condition; select the intended --run-id values')
        if s['status']!='complete' or cfg['epochs']!=expected_epochs or s['accepted_optimizer_steps']!=expected_steps:
            raise ValueError(f'Only complete {expected_epochs}-epoch {stage} runs qualify')
        if cfg['source_hashes']!=identity():raise ValueError('Study code differs from checkpoint source')
        if not (path.parent/'latest_checkpoint.pt').is_file():raise ValueError(f'Missing checkpoint: {path.parent}')
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
    p.add_argument('--stage',choices=['study','explore'],default='study',help='Explicitly select 64-epoch study or 16-epoch exploration')
    p.add_argument('--resume',action='store_true',help='Resume the exact frozen confirmation protocol without repeating completed models')
    a=p.parse_args()
    selected=select_runs(a.root,a.family,a.seeds,a.run_id,a.stage)
    epochs={'study':64,'explore':16}[a.stage]
    expected_steps=196*epochs
    print(f'Preflight: {len(selected)} local final checkpoints selected; {a.stage}, {epochs} epochs',flush=True)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    protocol=dict(protocol=PROTOCOL,families=a.family,seeds=a.seeds,
        stage=a.stage,epochs=epochs,examples=512,seed_offset=400000,overwrite_domain='held_out',
        device=a.device,selection='all final checkpoints; no best selection',
        checkpoint_selection={str(key):str(path) for key,(path,summary) in selected.items()})
    if a.resume:
        if json.loads((a.output/'confirmation_protocol.json').read_text())!=protocol:
            raise ValueError('Cannot resume a different confirmation protocol')
    else:atomic_write_json(a.output/'confirmation_protocol.json',protocol)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device)
    report=(json.loads((a.output/'confirmation_results.json').read_text()) if a.resume else
            dict(status='running',results=[],contrasts=[],device=str(device)))
    previous_wall=report.get('wall_seconds',0.) if a.resume else 0.
    report.update(status='running',contrasts=[]);report.pop('error',None)
    started=time.monotonic()
    try:
        for family in a.family:
            for seed in a.seeds:
                rows=[];paired=None;shared=None;base_parameters={}
                cfg=selected[(family,seed,'default','scalar')][1]['settings']
                split='held_out' if family=='overwrite' else 'confirmation'
                data=build_data(family,512,seed+400000,split,cfg['vocabulary'],cfg['length'])
                cached=a.output/f'{family}_seed{seed}_data.json'
                sample=a.output/'diagnostic_training_samples'/f'{family}_seed{seed}.npz'
                if a.resume and cached.exists() and sample.exists():
                    saved_data=json.loads(cached.read_text())
                    if not same_manifest(data.identity,saved_data['identity']):
                        raise ValueError('Resumed confirmation data identity differs')
                    overlaps=saved_data['overlap']
                    if set(overlaps)!={'train','development','previous_confirmation'}:
                        raise ValueError('Cached overlap audit incomplete')
                else:
                    overlaps={}
                    for name,count,offset,domain in [('train',100000,100000,'train'),('development',512,200000,'validation'),
                        ('previous_confirmation',512,300000,split)]:
                        old=build_data(family,count,seed+offset,domain,cfg['vocabulary'],cfg['length'])
                        if name=='train':
                            original=json.loads((selected[(family,seed,'default','scalar')][0].parent/'data_manifests.json').read_text())
                            if not same_manifest(old.identity,original['train']):
                                raise ValueError('Regenerated training data identity differs')
                            samples=a.output/'diagnostic_training_samples';samples.mkdir(exist_ok=True)
                            np.savez_compressed(sample,tokens=old.tokens[:512].numpy(),labels=old.labels[:512].numpy(),
                                erase_answers=old.erase_answers[:512].numpy())
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
                        existing=next((row for row in report['results'] if
                            (row['family'],row['seed'],row['initialization'],row['variant'])==
                            (family,seed,initialization,variant)),None)
                        if existing is not None:
                            if existing['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                                raise ValueError('Completed confirmation checkpoint changed')
                            rows.append(existing)
                            continue
                        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                        if saved['settings']!=summary['settings'] or saved['completed_steps']!=expected_steps or saved['completed_epochs']!=epochs:
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
                        print(f'{family}/seed{seed}/{initialization}/{variant}: '
                              f'{primary["correct"]}/{primary["total"]} ({primary["exact_query_accuracy"]:.4%})',flush=True)
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
        report['wall_seconds']=previous_wall+time.monotonic()-started
        atomic_write_json(a.output/'confirmation_results.json',report)


if __name__=='__main__':main()
