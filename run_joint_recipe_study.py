"""Isolated joint-initialization runner, forked from run_recipe_study.py.

Keep the legacy runner unchanged so its checkpoint source identities remain valid.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from artifacts import atomic_write_json
from recipe_calibration import qualify_calibrated, TARGETS
from joint_recipe_calibration import calibrate_joint as calibrate, SENSITIVITIES
from recipe_data import build_data, split_overlap, audit_mqar
from recipe_evaluation import evaluate
from recipe_model import RecipeLM

PROTOCOL='joint-timescale-sensitivity-v1'
VARIANTS=('scalar','channel-wise')


def identity():
    from run_published_recipe import source_identity
    names=(*source_identity().keys(),'recipe_calibration.py','run_recipe_study.py','run_joint_recipe_study.py','joint_recipe_calibration.py',
           'study_configs/calibrated_recipe_study.json','benchmark_kaggle_recipe.py','recurrence.py')
    return {name:hashlib.sha256(Path(name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in names}


def learning_summary(curve):
    def area(key):
        values=[row['answer_free'][key] for row in curve[:16]]
        return sum((a+b)/2 for a,b in zip(values,values[1:]))/15 if len(values)==16 else None
    def pass95(row):
        scores=row['answer_free']
        return all(scores[key] is None or scores[key]>=.95 for key in (
            'exact_query_accuracy','first_query_accuracy','later_query_accuracy',
            'latest_value_accuracy','unrelated_association_retention'))
    crossing=next((curve[i]['step'] for i in range(len(curve)-2)
                   if all(pass95(row) for row in curve[i:i+3])),None)
    return dict(accuracy_aulc_epochs1_16=area('exact_query_accuracy'),
                retention_aulc_epochs1_16=(area('unrelated_association_retention')
                    if curve[0]['answer_free']['unrelated_association_retention'] is not None else None),
                first_step_of_three_consecutive_95=crossing,
                threshold_95_right_censored=crossing is None,
                qualification_pass=len(curve)>=3 and all(all(row['answer_free'][key] is None or
                    row['answer_free'][key]>=.8 for key in ('exact_query_accuracy','first_query_accuracy',
                    'later_query_accuracy','latest_value_accuracy','unrelated_association_retention')) for row in curve[-3:]))


def worker(args):
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    cfg=dict(protocol=PROTOCOL,family=args.family[0],seed=args.seeds[0],variant=args.variant,
             initialization=args.initialization[0],stage=args.stage,epochs=args.epochs,scheduler_epochs=64,
             vocabulary=128 if args.cpu_smoke else 8192,length=32 if args.cpu_smoke else 64,
             width=16 if args.cpu_smoke else 128,train_examples=32 if args.cpu_smoke else 100000,
             validation_examples=16 if args.cpu_smoke else 512,effective_batch=512,microbatch=32,
             learning_rate=10**(-8/3),precision='float32',source_hashes=identity(),torch=str(torch.__version__),
             calibration_targets=list(TARGETS),sensitivity_targets=list(SENSITIVITIES),calibration_examples=16 if args.cpu_smoke else 256)
    atomic_write_json(args.output/'settings.json',cfg)
    family,seed=cfg['family'],cfg['seed']
    train=build_data(family,cfg['train_examples'],seed+100000,'train',cfg['vocabulary'],cfg['length'])
    dev=build_data(family,cfg['validation_examples'],seed+200000,'validation',cfg['vocabulary'],cfg['length'])
    audit=dict(train_development_content_overlap=split_overlap(train,dev))
    if family=='mqar': audit.update(audit_mqar(dev.tokens,dev.labels))
    else:
        from config import RetrievalConfig
        from retrieval import RetrievalDataset, evaluate_retrieval_baselines
        config=RetrievalConfig(vocabulary_size=cfg['vocabulary'],sequence_lengths=(cfg['length']+1,),
                              key_counts=(4,),query_distances=(8,),overwrite_counts=(1,),generator_version='mqar-overwrite-v4')
        data=RetrievalDataset(task='overwrite',examples=cfg['validation_examples'],config=config,seed=seed+200000,split='validation')
        frequency=RetrievalDataset(task='overwrite',examples=256,config=config,seed=seed+100000,split='train')
        audit.update(evaluate_retrieval_baselines(data,frequency_dataset=frequency))
        audit['passed']=(audit['status']=='complete' and audit['oracle_audit']['pass'] and
                         audit['counterfactual_query_key_audit']['pass'] and audit['shortcut_criterion']['all_cells_pass'])
    if not audit['passed'] or audit['train_development_content_overlap']:
        raise RuntimeError('Study data audit failed')
    atomic_write_json(args.output/'data_audit.json',audit)
    atomic_write_json(args.output/'data_manifests.json',dict(train=train.identity,development=dev.identity))
    model=RecipeLM(args.variant,cfg['vocabulary'],cfg['length'],cfg['width'],'chunked',seed).to(device)
    before={name:p.detach().clone() for name,p in model.named_parameters()}
    def fingerprint(items):
        digest=hashlib.sha256()
        for name,value in sorted(items):
            digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    atomic_write_json(args.output/'initialization_manifest.json',dict(
        base_parameter_sha256=fingerprint(before.items()),
        shared_parameter_sha256=fingerprint((name,value) for name,value in before.items()
            if not any(part in name for part in ('a_proj','A_log','dt_bias')))))
    calibration=dict(passed=True,initialization='default',wall_seconds=0.)
    if cfg['initialization']=='joint':
        calibration=calibrate(model,train.tokens[:cfg['calibration_examples']])
        if any(not torch.equal(before[name],p) for name,p in model.named_parameters() if not name.endswith(('dt_bias','A_log'))):
            raise RuntimeError('Joint calibration changed a non-memory parameter')
    del before
    atomic_write_json(args.output/'calibration.json',calibration)
    if cfg['initialization']=='joint':
        atomic_write_json(args.output/'calibrated_backend_qualification.json',
                          qualify_calibrated(model,train.tokens[:2]))
    optimizer=model.optimizer(cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=64,eta_min=0.)
    torch.manual_seed(seed+1000)
    setup=time.monotonic()-start
    curve=[];durations=[];accepted=0;validation_seconds=checkpoint_seconds=0.
    model.train()
    for epoch in range(args.epochs):
        for offset in range(0,len(train.tokens),512):
            if time.monotonic()-start>args.max_seconds-300:
                raise TimeoutError('Worker time reserve reached')
            tick=time.monotonic();optimizer.zero_grad(set_to_none=True)
            actual=min(512,len(train.tokens)-offset);last_loss=0.
            for micro in range(offset,offset+actual,32):
                stop=min(micro+32,offset+actual)
                loss=model.loss(train.tokens[micro:stop].to(device),train.labels[micro:stop].to(device))
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                weighted=loss*((stop-micro)/actual);weighted.backward();last_loss+=weighted.item()
            norm=torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None],foreach=True)
            if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradients')
            optimizer.step()
            if device.type=='cuda':torch.cuda.synchronize(device)
            durations.append(time.monotonic()-tick);accepted+=1
            if accepted==50:
                ordered=sorted(durations)
                conservative=max(statistics.mean(durations)*1.2,ordered[math.ceil(.95*len(ordered))-1])
                left=(math.ceil(len(train.tokens)/512)*args.epochs-accepted)*conservative
                if left+time.monotonic()-start+600>args.max_seconds:
                    raise TimeoutError('Measured training speed cannot fit the requested budget')
            if accepted%50==0:print(f'{family}/{cfg["initialization"]}/{args.variant}: {accepted} steps',flush=True)
        tick=time.monotonic()
        primary=evaluate(model,dev,device,batch=32)
        secondary=evaluate(model,dev,device,batch=32,history='teacher_forced') if family=='overwrite' else dict(primary,history_conditioning='teacher_forced')
        elapsed=time.monotonic()-tick;validation_seconds+=elapsed
        curve.append(dict(epoch=epoch+1,step=accepted,learning_rate=optimizer.param_groups[0]['lr'],
                          train_loss=last_loss,answer_free=primary,teacher_forced=secondary,validation_seconds=elapsed))
        atomic_write_json(args.output/'development_curve.json',curve)
        scheduler.step()
        tick=time.monotonic()
        checkpoint=args.output/'latest_checkpoint.pt'
        temporary=args.output/'checkpoint.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        completed_steps=accepted,completed_epochs=epoch+1,settings=cfg,
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),
                   temporary)
        temporary.replace(checkpoint)
        checkpoint_seconds+=time.monotonic()-tick
        print(json.dumps(curve[-1]),flush=True)
    expected=math.ceil(len(train.tokens)/512)*args.epochs
    if accepted!=expected:raise RuntimeError('Unexpected optimizer step count')
    atomic_write_json(args.output/'summary.json',dict(status='complete',settings=cfg,accepted_optimizer_steps=accepted,
        optimizer_step_seconds=durations,setup_seconds=setup,calibration_seconds=calibration['wall_seconds'],
        validation_seconds=validation_seconds,checkpoint_seconds=checkpoint_seconds,
        wall_seconds=time.monotonic()-start,gpu=None if args.cpu_smoke else torch.cuda.get_device_name(0),
        learning=learning_summary(curve),final_development=curve[-1],
        parameters=sum(p.numel() for p in model.parameters())))


def coordinator(args):
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    devices=args.devices.split(',')
    if not args.cpu_smoke:
        if not torch.cuda.is_available() or max(map(int,devices))>=torch.cuda.device_count():
            raise RuntimeError('Requested CUDA devices unavailable')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    protocol=dict(protocol=PROTOCOL,stage=args.stage,families=args.family,seeds=args.seeds,initializations=args.initialization,
        epochs=args.epochs,scheduler_epochs=64,targets=list(TARGETS),source_hashes=identity(),
        torch=str(torch.__version__),devices=devices,cpu_smoke=args.cpu_smoke,
        confirmation_policy='separate command after complete study; offset400000; never tune on confirmation')
    atomic_write_json(args.output/'study_protocol.json',protocol)
    queue=[(f,s,i) for s in args.seeds for f in args.family for i in args.initialization]
    status=dict(status='running',completed=[],deferred=[],protocol=protocol)
    try:
        if not args.cpu_smoke:
            from benchmark_kaggle_recipe import numerical_qualification
            atomic_write_json(args.output/'numerical_qualification.json',numerical_qualification(torch.device('cuda:'+devices[0]),16))
            torch.cuda.empty_cache()
        for index,(family,seed,initialization) in enumerate(queue):
            estimate=0 if args.cpu_smoke else 6000.
            remaining=args.max_seconds-(time.monotonic()-start)
            if estimate>remaining-300:
                status['deferred']=[dict(family=f,seed=s,initialization=i) for f,s,i in queue[index:]]
                break
            pair=args.output/family/f'seed_{seed}'/initialization
            pair.mkdir(parents=True)
            shutil.copy2(args.output/'study_protocol.json',pair/'study_protocol.json')
            if (args.output/'numerical_qualification.json').exists():
                shutil.copy2(args.output/'numerical_qualification.json',pair/'numerical_qualification.json')
            state=dict(status='running',family=family,seed=seed,initialization=initialization,stage=args.stage,
                source_hashes=identity(),torch=str(torch.__version__),cpu_smoke=args.cpu_smoke,
                hardware=[] if args.cpu_smoke else [torch.cuda.get_device_name(int(d)) for d in devices])
            processes=[];logs=[];tick=time.monotonic()
            try:
                for variant,device in zip(VARIANTS,devices if len(devices)==2 else devices*2):
                    log=(pair/(variant+'.log')).open('w',encoding='utf-8');logs.append(log)
                    command=[sys.executable,__file__,'--worker','--variant',variant,'--family',family,
                        '--seeds',str(seed),'--initialization',initialization,'--stage',args.stage,
                        '--output',str(pair/variant),'--max-seconds',str(int(remaining-60))]
                    if args.cpu_smoke:command.append('--cpu-smoke')
                    processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')))
                while any(p.poll() is None for p in processes):
                    if any(p.poll() not in (None,0) for p in processes):raise RuntimeError(f'Worker failed: {pair}')
                    if time.monotonic()-start>args.max_seconds-120:raise TimeoutError('Session reserve reached')
                    time.sleep(.5)
                if any(p.returncode for p in processes):raise RuntimeError('Worker failed; inspect pair logs')
                manifests=[json.loads((pair/v/'data_manifests.json').read_text()) for v in VARIANTS]
                if manifests[0]!=manifests[1]:raise RuntimeError('Paired data mismatch')
                shared=[json.loads((pair/v/'initialization_manifest.json').read_text())['shared_parameter_sha256'] for v in VARIANTS]
                if shared[0]!=shared[1]:raise RuntimeError('Paired shared backbone initialization mismatch')
                state.update(status='complete',qualification_pass=all(json.loads((pair/v/'summary.json').read_text())['learning']['qualification_pass'] for v in VARIANTS))
                status['completed'].append(str(pair.relative_to(args.output)))
            except BaseException as error:
                state.update(status='failed_or_incomplete',error=str(error));raise
            finally:
                for p in processes:
                    if p.poll() is None:p.terminate()
                for p in processes:
                    try:p.wait(timeout=10)
                    except subprocess.TimeoutExpired:p.kill();p.wait()
                for log in logs:log.close()
                state['wall_seconds']=time.monotonic()-tick
                atomic_write_json(pair/'pair_status.json',state)
                if args.publish:
                    from recipe_study_artifacts import publish_run
                    for variant in VARIANTS:
                        if (pair/variant).exists():
                            branch=publish_run(pair,variant,args.output.name,PROTOCOL)
                            print('Published:',branch,flush=True)
        status['status']='deferred' if status['deferred'] else 'complete'
        print(json.dumps(dict(status=status['status'],completed=status['completed'],deferred=status['deferred'])),flush=True)
    except BaseException as error:
        status.update(status='failed_or_incomplete',error=str(error));raise
    finally:
        status['wall_seconds']=time.monotonic()-start
        atomic_write_json(args.output/'session_status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',nargs='+',choices=['mqar','overwrite'],default=['mqar','overwrite'])
    group=p.add_mutually_exclusive_group()
    group.add_argument('--seeds',nargs='+',type=int)
    group.add_argument('--seed-count',type=int)
    p.add_argument('--start-seed',type=int,default=42)
    p.add_argument('--initialization',nargs='+',choices=['joint'],default=['joint'])
    p.add_argument('--stage',choices=['explore'],default='explore')
    p.add_argument('--devices',default='0',help='0 for two workers sharing GPU0, or 0,1 for one per GPU')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=39600)
    p.add_argument('--publish',action='store_true',default=False,help='Explicitly enable publication; disabled by default')
    p.add_argument('--no-publish',dest='publish',action='store_false')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--variant',choices=VARIANTS,help=argparse.SUPPRESS)
    p.add_argument('--cpu-smoke',action='store_true')
    p.add_argument('--plan-only',action='store_true',help='Print selected runs without loading a GPU or writing files')
    args=p.parse_args()
    if args.seed_count is not None and args.seed_count<1:p.error('Seed count must be positive')
    args.seeds=args.seeds or list(range(args.start_seed,args.start_seed+(args.seed_count or 3)))
    if any(s<0 for s in args.seeds) or len(set(args.seeds))!=len(args.seeds):p.error('Seeds must be unique nonnegative integers')
    if any(len(set(x))!=len(x) for x in [args.family,args.initialization]):p.error('Selections must be unique')
    if not 900<args.max_seconds<=39600:p.error('Workload ceiling must be 901..39600 seconds')
    if len(args.devices.split(',')) not in (1,2) or any(not d.isdigit() for d in args.devices.split(',')):p.error('Use --devices 0 or 0,1')
    if args.worker and (not args.variant or len(args.family)!=1 or len(args.seeds)!=1 or len(args.initialization)!=1):p.error('Worker requires one complete condition')
    args.epochs=2 if args.cpu_smoke else 16
    if args.cpu_smoke:args.publish=False
    args.output=args.output.resolve()
    if args.plan_only:
        print(json.dumps(dict(stage=args.stage,families=args.family,seeds=args.seeds,initialization=args.initialization,
            models=2*len(args.family)*len(args.seeds)*len(args.initialization),epochs=args.epochs,
            scheduler_epochs=64,publish=args.publish,devices=args.devices,output=str(args.output)),indent=2))
        return
    if args.worker:worker(args)
    else:coordinator(args)


if __name__=='__main__':main()
