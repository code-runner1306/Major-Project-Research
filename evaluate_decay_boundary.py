"""Held-out decay interventions and paired update tests; read-only checkpoints."""
import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from artifacts import atomic_write_json
from confirm_recipe_study import select_runs,same_manifest
from recipe_data import RecipeData,build_data,manifest,split_overlap
from recipe_evaluation import evaluate
from recipe_model import RecipeLM,RecipeGDN
from stronger_gate_tests import intervention
from stress_recipe_checkpoints import StressCase,make_mqar,make_overwrite

PROTOCOL='dynamic-decay-boundary-discovery-v1'
MQAR=[('baseline',64,4),('load8',64,8),('load12',64,12),
      ('length128',128,4),('length128_load8',128,8),('length128_load12',128,12)]
OVERWRITE=[('baseline',4,8,1),('load8',8,8,1),('distance24',4,24,1),
           ('rewrites2',4,8,2),('rewrites3',4,8,3),('rewrites4',4,8,4)]
PAIRS=('obsolete_prefix','latest_value','rename_key')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    names=['evaluate_decay_boundary.py','stronger_gate_tests.py','stress_recipe_checkpoints.py',
           'confirm_recipe_study.py','recipe_evaluation.py','recipe_data.py','retrieval.py',
           'recipe_model.py','config.py','experimental_chunked_recurrence.py','recurrence.py']
    return {name:hashlib.sha256(Path(name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in names}


def select_checkpoints(args):
    selected=select_runs(args.root,args.family,args.seeds,args.run_id,'explore')
    if args.joint_root:
        from run_joint_recipe_study import identity
        joint={}
        for path in args.joint_root.rglob('summary.json'):
            summary=json.loads(path.read_text());cfg=summary.get('settings',{})
            if cfg.get('protocol')!='joint-timescale-sensitivity-v1':continue
            if cfg.get('family') not in args.family or cfg.get('seed') not in args.seeds:continue
            if cfg.get('stage')!='explore' or cfg.get('initialization')!='joint':continue
            if args.joint_run_id and not any(x in path.parts for x in args.joint_run_id):continue
            key=(cfg['family'],cfg['seed'],'joint',cfg['variant'])
            if key in joint:raise ValueError('Duplicate joint condition; use --joint-run-id')
            if (summary.get('status')!='complete' or cfg.get('epochs')!=16
                    or summary.get('accepted_optimizer_steps')!=3136 or cfg.get('source_hashes')!=identity()):
                raise ValueError(f'Incomplete or incompatible joint checkpoint: {path}')
            joint[key]=(path,summary)
        expected={(f,s,'joint',v) for f in args.family for s in args.seeds for v in ('scalar','channel-wise')}
        if set(joint)!=expected:raise ValueError(f'Incomplete joint matrix: {sorted(expected-set(joint))}')
        selected.update(joint)
    paired={};backbones={}
    for key,(path,summary) in sorted(selected.items()):
        cfg=summary['settings'];checkpoint=path.parent/'latest_checkpoint.pt'
        if not checkpoint.is_file():raise ValueError(f'Missing checkpoint: {checkpoint}')
        if (cfg['vocabulary'],cfg['length'],cfg['width'],cfg['train_examples'],cfg['validation_examples'],
            cfg['effective_batch'],cfg['microbatch'],cfg['scheduler_epochs'],cfg['precision'])!=(8192,64,128,100000,512,512,32,64,'float32'):
            raise ValueError(f'Unexpected training recipe: {path}')
        if abs(cfg['learning_rate']-10**(-8/3))>1e-12:raise ValueError('Learning-rate mismatch')
        data=json.loads((path.parent/'data_manifests.json').read_text())
        initial=json.loads((path.parent/'initialization_manifest.json').read_text())
        pair=key[:2]
        if pair in paired and not same_manifest(paired[pair],data):raise ValueError('Paired training/development data differ')
        if pair in backbones and backbones[pair]!=initial['shared_parameter_sha256']:raise ValueError('Shared backbone initialization differs')
        paired[pair]=data;backbones[pair]=initial['shared_parameter_sha256']
    if args.cpu_smoke:selected={k:v for k,v in selected.items() if k[2]=='default'}
    return selected


def audit_overwrite(case):
    """Scan each query independently, ignoring filler and previous answer tokens."""
    data=case.data
    for row in range(len(data.tokens)):
        sequence=data.tokens[row].tolist()
        for position in (data.labels[row]!=-100).nonzero().flatten().tolist():
            key=sequence[position];query=position-1
            if sequence[query]!=0:raise ValueError('Missing query marker')
            values=[sequence[p+1] for p in range(query-1) if sequence[p]==key and sequence[p+1]>=4096]
            if not values or values[-1]!=int(data.labels[row,position]):raise ValueError('Latest-write oracle failed')
            if int(data.kinds[row,position])==1:
                recorded=case.stale_values[row,position]
                if values[:-1]!=recorded[recorded>=0].tolist():raise ValueError('Stale-value order differs from oracle')
    return True


def paired_case(base,kind,seed):
    data=base.data
    tokens,labels,kinds,stale,erase=[x.clone() for x in (data.tokens,data.labels,data.kinds,data.stale,data.erase_answers)]
    all_stale=base.stale_values.clone();rng=random.Random(seed)
    for row in range(len(tokens)):
        position=int((kinds[row]==1).nonzero().item());key=int(tokens[row,position]);query=position-1
        writes=[p+1 for p in range(query-1) if int(tokens[row,p])==key and int(tokens[row,p+1])>=4096]
        if len(writes)<2:raise ValueError('Paired update tests require an overwritten key')
        used=set(tokens[row].tolist())|set(labels[row][labels[row]>=0].tolist())
        low,high=(2,4096) if kind=='rename_key' else (4096,8192)
        candidate=rng.randrange(low,high)
        while candidate in used:candidate=low+(candidate-low+1)%(high-low)
        if kind=='rename_key':tokens[row,tokens[row]==key]=candidate
        elif kind=='latest_value':
            tokens[row,writes[-1]]=candidate;labels[row,position]=candidate
            if position+1<tokens.shape[1]:tokens[row,position+1]=candidate
        elif kind=='obsolete_prefix':
            tokens[row,writes[0]]=candidate;all_stale[row,position,0]=candidate
        else:raise ValueError('Unknown pair type')
        prior=all_stale[row,position];stale[row,position]=prior[prior>=0][-1]
    result=StressCase(RecipeData(tokens,labels,kinds,stale,erase,
        manifest(tokens,labels,phase='discovery',pair=kind,base_sha256=data.identity['sha256'],seed=seed)),all_stale,
        dict(passed=True,scope='paired oracle and target-change checks'))
    audit_overwrite(result)
    latest=kinds==1
    if kind=='latest_value':
        if not torch.all(labels[latest]!=data.labels[latest]):raise ValueError('Latest targets did not change')
        if not torch.equal(labels[~latest],data.labels[~latest]):raise ValueError('Unrelated labels changed')
    elif not torch.equal(labels,data.labels):raise ValueError('Invariant pair changed labels')
    return result


@torch.no_grad()
def fit_constants(model,data,device,examples,batch):
    values={};handles=[];was_training=model.training
    def capture(layer,inputs):
        alpha=layer.decay(inputs[0]).detach().double()
        part=alpha.sum((0,1)).cpu();count=alpha.shape[0]*alpha.shape[1]
        if layer.layer_idx not in values:values[layer.layer_idx]=[part,count]
        else:values[layer.layer_idx][0]+=part;values[layer.layer_idx][1]+=count
    tokens=data.tokens[:examples].clone();tokens[data.erase_answers[:examples]]=1
    try:
        model.eval()
        for layer in model.modules():
            if isinstance(layer,RecipeGDN):handles.append(layer.register_forward_pre_hook(capture))
        for start in range(0,len(tokens),batch):model.network.backbone(tokens[start:start+batch].to(device))
    finally:
        for handle in handles:handle.remove()
        model.train(was_training)
    return {layer:(total/count).float() for layer,(total,count) in values.items()}


def score(model,case,device,batch):
    predictions=[];original=model.scored_logits;had='scored_logits' in model.__dict__
    def capture(tokens,mask):
        logits=original(tokens,mask);predictions.append(logits.argmax(-1).detach().cpu());return logits
    model.scored_logits=capture
    try:metrics=evaluate(model,case.data,device,batch=batch)
    finally:
        if had:model.scored_logits=original
        else:delattr(model,'scored_logits')
    prediction=torch.cat(predictions);mask=case.data.labels!=-100
    latest=case.data.kinds[mask]==1;prior=case.stale_values[mask]
    hits=((prediction[:,None]==prior)&(prior>=0)).any(-1)&latest
    metrics['stale_value_error_rate']=float(hits.sum()/latest.sum()) if latest.any() else None
    if not np.isfinite(metrics['loss']):raise FloatingPointError('Nonfinite evaluation loss')
    return metrics,prediction


@torch.no_grad()
def qualify_forward(model,case,device):
    """Check the active intervention on actual examples against serial recurrence."""
    tokens=case.data.tokens[:2].clone();tokens[case.data.erase_answers[:2]]=1
    mask=case.data.labels[:2]!=-100
    layers=[m for m in model.modules() if isinstance(m,RecipeGDN)]
    backends=[m.backend for m in layers]
    modes={module:module.training for module in model.modules()}
    try:
        model.eval()
        for layer in layers:layer.backend='chunked'
        chunked=model.scored_logits(tokens.to(device),mask.to(device)).detach().cpu()
        for layer in layers:layer.backend='reference'
        reference=model.scored_logits(tokens.to(device),mask.to(device)).detach().cpu()
        torch.testing.assert_close(chunked,reference,rtol=3e-4,atol=3e-4)
        return dict(passed=True,examples=len(tokens),rtol=3e-4,atol=3e-4,
                    max_logit_absolute_error=float((chunked-reference).abs().max()))
    finally:
        for layer,backend in zip(layers,backends):layer.backend=backend
        for module,training in modes.items():module.training=training


def pair_metrics(base,changed,base_predictions,new_predictions,kind):
    mask=base.data.labels!=-100;latest=base.data.kinds[mask]==1;unrelated=base.data.kinds[mask]==2
    correct0=base_predictions==base.data.labels[mask];correct1=new_predictions==changed.data.labels[mask]
    def mean(values,group):return float(values[group].float().mean()) if group.any() else None
    return dict(pair_kind=kind,latest_joint_correct=mean(correct0&correct1,latest),
        latest_changed_fraction=mean(base_predictions!=new_predictions,latest),
        latest_transformed_accuracy=mean(correct1,latest),unrelated_joint_correct=mean(correct0&correct1,unrelated),
        unrelated_changed_fraction=mean(base_predictions!=new_predictions,unrelated),
        latest_correct_after_given_correct_before=mean(correct1,latest&correct0),
        latest_baseline_correct_count=int((latest&correct0).sum()),
        interpretation='Agreement alone is not correctness; latest_value expects change, other pairs expect invariance')


def cases_for(family,seed,examples):
    cases={}
    for cell in MQAR if family=='mqar' else OVERWRITE:
        cases[cell[0]]=(make_mqar(examples,seed+700000,cell[1],cell[2]) if family=='mqar' else
            make_overwrite(examples,seed+700000,cell[1],cell[2],cell[3],seed+100000))
        if family=='overwrite':audit_overwrite(cases[cell[0]])
    if family=='overwrite':
        for index,kind in enumerate(PAIRS):cases['pair_'+kind]=paired_case(cases['rewrites3'],kind,seed+710000+index*1000)
    return cases


def source_training(family,seed,args,selected):
    if args.cpu_smoke:
        path=args.training_cache/f'{family}_seed{seed}.npz'
        with np.load(path) as cached:
            tokens=torch.from_numpy(cached['tokens'].copy());labels=torch.from_numpy(cached['labels'].copy())
            erase=torch.from_numpy(cached['erase_answers'].copy())
        return RecipeData(tokens,labels,torch.full_like(labels,-1),torch.full_like(labels,-1),erase,
            manifest(tokens,labels,source='cached exact training prefix; smoke only'))
    train=build_data(family,100000,seed+100000,'train',8192,64)
    path=next(path for (f,s,i,v),(path,_) in selected.items() if (f,s)==(family,seed))
    expected=json.loads((path.parent/'data_manifests.json').read_text())['train']
    if not same_manifest(train.identity,expected):raise ValueError('Regenerated training pool identity differs')
    return train


def summarize(report,output):
    flat=[];pairs=[]
    for row in report['results']:
        flat.append({**{k:row[k] for k in ('family','seed','initialization','variant','intervention','cell')},
            **row['metrics'],'wall_seconds':row['wall_seconds']})
        if 'paired_metrics' in row:
            pairs.append({**{k:row[k] for k in ('family','seed','initialization','variant','intervention','cell')},**row['paired_metrics']})
    if flat:
        with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    if pairs:
        with (output/'paired_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    index={(r['family'],r['seed'],r['initialization'],r['variant'],r['intervention'],r['cell']):r for r in report['results']}
    contrasts=[]
    for family,stress,metric in [('mqar','load8','exact_query_accuracy'),('overwrite','rewrites3','latest_value_accuracy')]:
        for seed in report['protocol']['seeds']:
            for init in report['protocol']['initializations']:
                keys=[(family,seed,init,'channel-wise',mode,cell) for cell in ('baseline',stress) for mode in ('original','static')]
                if not all(k in index for k in keys):continue
                baseline=index[keys[0]]['metrics'][metric]-index[keys[1]]['metrics'][metric]
                dynamic=index[keys[2]]['metrics'][metric];static=index[keys[3]]['metrics'][metric]
                gap=dynamic-static
                contrasts.append(dict(family=family,seed=seed,initialization=init,baseline_gap=baseline,
                    stress_gap=gap,stress_specific_gap=gap-baseline,
                    seed_operational_reliance=gap>=.05 and abs(baseline)<=.02,
                    reliable_dynamic_only=dynamic>=.95 and static<.90,
                    interpretation='Per-seed descriptive rule; require all three seeds, not a significance test'))
    atomic_write_json(output/'contrasts.json',contrasts)
    lines=['# Decay boundary evaluation', '',f"Status: {report['status']}",
        f"Completed evaluations: {len(report['results'])}",f"Total recorded wall seconds: {report.get('wall_seconds',0):.1f}",
        '', 'See metrics.csv, contrasts.json, results.json, audits/, fitted_gates/, and predictions/.',
        'No training or publication was performed. This is discovery evidence; confirmation offset 900000 is reserved.',
        'Static interventions measure checkpoint reliance, not separately trained architecture superiority.',
        'Paired-test agreement must be read alongside joint correctness.',
        'Inspect all per-seed contrasts before drawing conclusions.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(args):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    selected=select_checkpoints(args)
    selection=[dict(family=k[0],seed=k[1],initialization=k[2],variant=k[3],
        path=str(p.parent/'latest_checkpoint.pt'),sha256=sha(p.parent/'latest_checkpoint.pt')) for k,(p,s) in sorted(selected.items())]
    protocol=dict(protocol=PROTOCOL,phase='discovery',families=args.family,seeds=args.seeds,
        initializations=sorted({k[2] for k in selected}),examples=args.examples,fit_examples=args.fit_examples,
        batch=args.batch,seed_offset=700000,reserved_confirmation_offset=900000,
        mqar_cells=MQAR,overwrite_cells=OVERWRITE,paired_tests=PAIRS,
        cpu_smoke=args.cpu_smoke,device='cpu' if args.cpu_smoke else 'cuda:0',publish=False,
        checkpoint_selection=selection,evaluator_sources=source_identity())
    executions=sum((1 if k[3]=='scalar' else 3)*(6 if k[0]=='mqar' else 9) for k in selected)
    if args.plan_only:
        print(json.dumps(dict(protocol=PROTOCOL,models=len(selected),evaluations=executions,examples=args.examples,
            device=protocol['device'],publish=False,output=str(args.output)),indent=2));return
    device=torch.device(protocol['device'])
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; production evaluation requires the A6000 CUDA environment')
    if args.resume:
        if not same_manifest(json.loads((args.output/'protocol.json').read_text()),protocol):raise ValueError('Resume protocol/checkpoint/source mismatch')
        report=json.loads((args.output/'results.json').read_text())
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        atomic_write_json(args.output/'protocol.json',protocol)
        report=dict(status='running',protocol=protocol,results=[],audits=[],wall_seconds=0.)
    for name in ('audits','predictions','fitted_gates'):(args.output/name).mkdir(exist_ok=True)
    atomic_write_json(args.output/'environment.json',dict(torch=str(torch.__version__),cuda_build=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,device=str(device),workers=1,publish=False))
    previous=report.get('wall_seconds',0.);started=time.monotonic()
    report.update(status='running');report.pop('error',None)
    done={(r['family'],r['seed'],r['initialization'],r['variant'],r['intervention'],r['cell']) for r in report['results']}
    if len(done)!=len(report['results']):raise ValueError('Duplicate completed evaluation records')
    existing={(r['family'],r['seed'],r['initialization'],r['variant'],r['intervention'],r['cell']):r for r in report['results']}
    try:
        if device.type=='cuda':print('GPU:',torch.cuda.get_device_name(0),flush=True)
        print(f'{len(selected)} models; {executions} evaluations; no training; publish=false',flush=True)
        for family in args.family:
            for seed in args.seeds:
                cases=cases_for(family,seed,args.examples)
                train=source_training(family,seed,args,selected)
                audit=dict(family=family,seed=seed,smoke_only=args.cpu_smoke,cases={})
                for cell,case in cases.items():
                    overlap=split_overlap(train,case.data)
                    if overlap:raise ValueError(f'Train overlap in {cell}')
                    audit['cases'][cell]=dict(identity=case.data.identity,train_overlap=overlap,audit=case.audit)
                # Distinct development and already observed confirmation examples.
                if not args.cpu_smoke:
                    for offset,split in [(200000,'validation'),(300000,'held_out'),(400000,'held_out')]:
                        old=build_data(family,512,seed+offset,split,8192,64)
                        for cell,case in cases.items():
                            if split_overlap(old,case.data):raise ValueError('Discovery overlaps a previous evaluation')
                        del old
                audit_path=args.output/'audits'/f'{family}_seed{seed}.json'
                if args.resume and audit_path.exists() and not same_manifest(json.loads(audit_path.read_text()),audit):raise ValueError('Resumed data audits changed')
                atomic_write_json(audit_path,audit)
                for (f,s,init,variant),(path,summary) in sorted(selected.items()):
                    if (f,s)!=(family,seed):continue
                    model=RecipeLM(variant,8192,64,128,'chunked',seed)
                    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                        saved=torch.load(path.parent/'latest_checkpoint.pt',map_location='cpu',weights_only=True)
                    if not same_manifest(saved['settings'],summary['settings']) or saved['completed_steps']!=3136 or saved['completed_epochs']!=16:raise ValueError('Checkpoint content differs from summary')
                    model.load_state_dict(saved['model'],strict=True);del saved
                    model.to(device).eval()
                    original={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
                    means=fit_constants(model,train,device,args.fit_examples,args.batch) if variant=='channel-wise' else None
                    model_name=f'{family}_seed{seed}_{init}_{variant}'
                    if means:atomic_write_json(args.output/'fitted_gates'/f'{model_name}.json',dict(training_identity=train.identity,examples=args.fit_examples,means={k:v.tolist() for k,v in means.items()}))
                    for mode in ('original','static','mean') if means else ('original',):
                        with nullcontext() if mode=='original' else intervention(model,mode,means=means):
                            proof=qualify_forward(model,cases['baseline'],device)
                        atomic_write_json(args.output/'audits'/f'{model_name}_{mode}_numerical.json',proof)
                        base_predictions=None
                        for cell,case in cases.items():
                            key=(family,seed,init,variant,mode,cell)
                            prediction_path=args.output/'predictions'/f'{model_name}_{mode}_{cell}.npz'
                            if key in done:
                                if not prediction_path.is_file():raise ValueError('Completed result is missing raw predictions')
                                if sha(prediction_path)!=existing[key]['prediction_sha256']:raise ValueError('Completed raw predictions changed')
                                with np.load(prediction_path) as cached:
                                    predictions=torch.from_numpy(cached['predictions'].copy())
                                    if not np.array_equal(cached['labels'],case.data.labels[case.data.labels!=-100].numpy()):raise ValueError('Resumed prediction labels differ')
                            else:
                                tick=time.monotonic()
                                with nullcontext() if mode=='original' else intervention(model,mode,means=means):
                                    metrics,predictions=score(model,case,device,args.batch)
                                if device.type=='cuda':torch.cuda.synchronize()
                                row=dict(family=family,seed=seed,initialization=init,variant=variant,
                                    intervention=mode,cell=cell,metrics=metrics,wall_seconds=time.monotonic()-tick)
                                if cell.startswith('pair_'):
                                    if base_predictions is None:raise ValueError('Pair baseline missing')
                                    row['paired_metrics']=pair_metrics(cases['rewrites3'],case,base_predictions,predictions,cell[5:])
                                mask=case.data.labels!=-100
                                np.savez_compressed(prediction_path,predictions=predictions.numpy(),labels=case.data.labels[mask].numpy(),
                                    kinds=case.data.kinds[mask].numpy(),example_index=mask.nonzero()[:,0].numpy())
                                row['prediction_sha256']=sha(prediction_path)
                                report['results'].append(row);done.add(key)
                                report['wall_seconds']=previous+time.monotonic()-started
                                atomic_write_json(args.output/'results.json',report)
                                print(f'{len(done)}/{executions}: {model_name}/{mode}/{cell} accuracy={metrics["exact_query_accuracy"]:.4f}',flush=True)
                            if cell=='rewrites3':base_predictions=predictions
                    if any(not torch.equal(original[n],p.detach().cpu()) for n,p in model.named_parameters()):raise ValueError('Evaluation modified checkpoint parameters')
                    del model,original
                    if device.type=='cuda':torch.cuda.empty_cache()
                del train,cases
        report['status']='complete'
    except BaseException as error:
        report.update(status='failed_or_interrupted',error=str(error));raise
    finally:
        report['wall_seconds']=previous+time.monotonic()-started
        atomic_write_json(args.output/'results.json',report);summarize(report,args.output)
    print('Complete:',args.output/'REPORT.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('output/calibrated_recipe_study/matched-decay-v1'))
    p.add_argument('--joint-root',type=Path);p.add_argument('--run-id',nargs='*',default=[]);p.add_argument('--joint-run-id',nargs='*',default=[])
    p.add_argument('--family',nargs='+',choices=['mqar','overwrite'],default=['mqar','overwrite'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,default=512);p.add_argument('--fit-examples',type=int,default=256)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--resume',action='store_true');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--cpu-smoke',action='store_true',help='Infrastructure check using cached training prefix; not scientific evidence')
    p.add_argument('--training-cache',type=Path,default=Path('research_results/three_seed_confirmation_20261117/diagnostic_training_samples'))
    args=p.parse_args()
    if any(len(set(x))!=len(x) for x in (args.family,args.seeds)):p.error('Selections must be unique')
    if any(s<0 for s in args.seeds) or min(args.examples,args.fit_examples,args.batch)<1 or args.fit_examples>512:p.error('Invalid example/batch/seed selection')
    if args.cpu_smoke:
        args.examples=min(args.examples,2);args.fit_examples=min(args.fit_examples,2);args.batch=2
        if args.joint_root:p.error('CPU smoke uses legacy default models only')
    args.output=args.output.resolve();args.root=args.root.resolve()
    run(args)


if __name__=='__main__':main()
