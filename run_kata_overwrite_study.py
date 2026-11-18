"""KATA-style repeated-key overwrite study (protocol kata-overwrite-v1), forked from run_layout_fix_study.py.

Trains default-initialization GDN models on kata_overwrite_data.py: every key is written exactly twice, as in the
published KATA overwrite task. The training loop, optimizer, schedule, batch sizes, model and seeds are unchanged
from the earlier studies; only the training data differ. Evaluation is a separate script (eval_kata_overwrite.py).
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
from kata_overwrite_data import ARMS, build_kata_data, oracle_audit
from recipe_data import split_overlap
from recipe_evaluation import evaluate
from recipe_model import RecipeLM

PROTOCOL='kata-overwrite-v1'
DEFAULT_ARMS=('kata_twice',)
PREDICTIONS={
    'K1_learned':'In-distribution (every key written twice) latest-value accuracy is at least 95% for every model.',
    'K2_rewrite_shortcut':'A model shows the rewrite shortcut if accuracy on the target key in target_three is at '
                          'least 20 points below in_distribution. Confirmed if at least 2 of 3 models show it.',
    'K3_error_identity':'Pooled over models, at least 50% of target-key errors in target_three and target_four '
                        'return the second write of that key.',
    'K4_descriptive':'target_once, all_once and all_three are reported without a threshold.',
    'evaluation':'eval_kata_overwrite.py at seed offset 1700000 after all training is complete; no tuning on it.'}
VARIANTS=('scalar','channel-wise')


def identity():
    from run_published_recipe import source_identity
    names=(*source_identity().keys(),'run_kata_overwrite_study.py','kata_overwrite_data.py','retrieval.py','config.py',
           'benchmark_kaggle_recipe.py','recurrence.py')
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
    cfg=dict(protocol=PROTOCOL,family='kata_overwrite',arm=args.arm[0],seed=args.seeds[0],variant=args.variant,
             initialization='default',stage=args.stage,epochs=args.epochs,scheduler_epochs=64,
             vocabulary=128 if args.cpu_smoke else 8192,length=32 if args.cpu_smoke else 64,
             width=16 if args.cpu_smoke else 128,train_examples=32 if args.cpu_smoke else 100000,
             validation_examples=16 if args.cpu_smoke else 512,effective_batch=512,microbatch=32,
             learning_rate=10**(-8/3),precision='float32',source_hashes=identity(),torch=str(torch.__version__),
             arm_definition=ARMS[args.arm[0]])
    atomic_write_json(args.output/'settings.json',cfg)
    family,seed,arm=cfg['family'],cfg['seed'],cfg['arm']
    train=build_kata_data(cfg['train_examples'],seed+100000,'train',cfg['vocabulary'],cfg['length'])
    dev=build_kata_data(cfg['validation_examples'],seed+200000,'validation',cfg['vocabulary'],cfg['length'])
    arm_dev=dev
    audit=dict(train_development_content_overlap=split_overlap(train,dev),
               development_oracle=oracle_audit(dev,cfg['vocabulary']),
               train_oracle_first_2000=oracle_audit(type(train)(train.tokens[:2000],train.labels[:2000],None,None,None,None),cfg['vocabulary']))
    audit['passed']=audit['development_oracle']['passed'] and audit['train_oracle_first_2000']['passed']
    if not audit['passed'] or audit['train_development_content_overlap']:
        raise RuntimeError('Study data audit failed')
    atomic_write_json(args.output/'data_audit.json',audit)
    atomic_write_json(args.output/'data_manifests.json',dict(train=train.identity,development=dev.identity,
                                                             arm_development=arm_dev.identity))
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
    del before
    atomic_write_json(args.output/'calibration.json',calibration)
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
            if accepted%50==0:print(f'{arm}/{args.variant}: {accepted} steps',flush=True)
        tick=time.monotonic()
        primary=evaluate(model,dev,device,batch=32)
        secondary=evaluate(model,dev,device,batch=32,history='teacher_forced')
        arm_primary=primary
        elapsed=time.monotonic()-tick;validation_seconds+=elapsed
        curve.append(dict(epoch=epoch+1,step=accepted,learning_rate=optimizer.param_groups[0]['lr'],
                          train_loss=last_loss,answer_free=primary,teacher_forced=secondary,
                          arm_answer_free=arm_primary,validation_seconds=elapsed))
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
    protocol=dict(protocol=PROTOCOL,stage=args.stage,arms=args.arm,arm_definitions={a:ARMS[a] for a in args.arm},
        seeds=args.seeds,variants=args.variants,initialization='default',predictions=PREDICTIONS,
        epochs=args.epochs,scheduler_epochs=64,source_hashes=identity(),
        torch=str(torch.__version__),devices=devices,cpu_smoke=args.cpu_smoke,
        confirmation_policy='eval_kata_overwrite.py after complete study at offset 1700000; never tune on it')
    atomic_write_json(args.output/'study_protocol.json',protocol)
    queue=[(a,s) for a in args.arm for s in args.seeds]
    status=dict(status='running',completed=[],deferred=[],protocol=protocol)
    try:
        if not args.cpu_smoke:
            from benchmark_kaggle_recipe import numerical_qualification
            atomic_write_json(args.output/'numerical_qualification.json',numerical_qualification(torch.device('cuda:'+devices[0]),16))
            torch.cuda.empty_cache()
        for index,(arm,seed) in enumerate(queue):
            estimate=0 if args.cpu_smoke else 4200.
            remaining=args.max_seconds-(time.monotonic()-start)
            if estimate>remaining-300:
                status['deferred']=[dict(arm=a,seed=s) for a,s in queue[index:]]
                break
            pair=args.output/arm/f'seed_{seed}'
            pair.mkdir(parents=True)
            shutil.copy2(args.output/'study_protocol.json',pair/'study_protocol.json')
            if (args.output/'numerical_qualification.json').exists():
                shutil.copy2(args.output/'numerical_qualification.json',pair/'numerical_qualification.json')
            state=dict(status='running',arm=arm,seed=seed,initialization='default',stage=args.stage,
                source_hashes=identity(),torch=str(torch.__version__),cpu_smoke=args.cpu_smoke,
                hardware=[] if args.cpu_smoke else [torch.cuda.get_device_name(int(d)) for d in devices])
            processes=[];logs=[];tick=time.monotonic()
            try:
                # Both variants: one worker each (one per GPU if two devices). One variant: a single worker on
                # the first device, so a notebook with one T4 can train its share of the study on its own.
                placements=(zip(args.variants,devices if len(devices)==2 else devices*2) if len(args.variants)==2
                            else [(args.variants[0],devices[0])])
                for variant,device in placements:
                    log=(pair/(variant+'.log')).open('w',encoding='utf-8');logs.append(log)
                    command=[sys.executable,__file__,'--worker','--variant',variant,'--arm',arm,
                        '--seeds',str(seed),'--stage',args.stage,
                        '--output',str(pair/variant),'--max-seconds',str(int(remaining-60))]
                    if args.cpu_smoke:command.append('--cpu-smoke')
                    processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')))
                while any(p.poll() is None for p in processes):
                    if any(p.poll() not in (None,0) for p in processes):raise RuntimeError(f'Worker failed: {pair}')
                    if time.monotonic()-start>args.max_seconds-120:raise TimeoutError('Session reserve reached')
                    time.sleep(.5)
                if any(p.returncode for p in processes):raise RuntimeError('Worker failed; inspect pair logs')
                if len(args.variants)==2:
                    manifests=[json.loads((pair/v/'data_manifests.json').read_text()) for v in VARIANTS]
                    if manifests[0]!=manifests[1]:raise RuntimeError('Paired data mismatch')
                    shared=[json.loads((pair/v/'initialization_manifest.json').read_text())['shared_parameter_sha256'] for v in VARIANTS]
                    if shared[0]!=shared[1]:raise RuntimeError('Paired shared backbone initialization mismatch')
                else:
                    state['pairing_check']='deferred: the other variant trains in a separate session; compare data and shared-initialization hashes when evaluating'
                state.update(status='complete',variants=args.variants,qualification_pass=all(json.loads((pair/v/'summary.json').read_text())['learning']['qualification_pass'] for v in args.variants))
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
        status['status']='deferred' if status['deferred'] else 'complete'
        print(json.dumps(dict(status=status['status'],completed=status['completed'],deferred=status['deferred'])),flush=True)
    except BaseException as error:
        status.update(status='failed_or_incomplete',error=str(error));raise
    finally:
        status['wall_seconds']=time.monotonic()-start
        atomic_write_json(args.output/'session_status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',nargs='+',choices=list(ARMS),default=list(DEFAULT_ARMS))
    group=p.add_mutually_exclusive_group()
    group.add_argument('--seeds',nargs='+',type=int)
    group.add_argument('--seed-count',type=int)
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS),
                   help='Train one variant only to split a study across separate single-GPU sessions')
    p.add_argument('--start-seed',type=int,default=42)
    p.add_argument('--stage',choices=['explore'],default='explore')
    p.add_argument('--devices',default='0',help='0 for two workers sharing GPU0, or 0,1 for one per GPU')
    p.add_argument('--output',type=Path)
    p.add_argument('--max-seconds',type=int,default=39600)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--variant',choices=VARIANTS,help=argparse.SUPPRESS)
    p.add_argument('--cpu-smoke',action='store_true')
    p.add_argument('--plan-only',action='store_true',help='Print selected runs without loading a GPU or writing files')

    args=p.parse_args()
    if args.seed_count is not None and args.seed_count<1:p.error('Seed count must be positive')
    args.seeds=args.seeds or list(range(args.start_seed,args.start_seed+(args.seed_count or 3)))
    if any(s<0 for s in args.seeds) or len(set(args.seeds))!=len(args.seeds):p.error('Seeds must be unique nonnegative integers')
    if len(set(args.arm))!=len(args.arm):p.error('Arms must be unique')
    if len(set(args.variants))!=len(args.variants):p.error('Variants must be unique')
    args.variants=[v for v in VARIANTS if v in args.variants]
    if not 900<args.max_seconds<=39600:p.error('Workload ceiling must be 901..39600 seconds')
    if len(args.devices.split(',')) not in (1,2) or any(not d.isdigit() for d in args.devices.split(',')):p.error('Use --devices 0 or 0,1')
    if args.worker and (not args.variant or len(args.arm)!=1 or len(args.seeds)!=1):p.error('Worker requires one complete condition')
    args.epochs=2 if args.cpu_smoke else 16
    if args.output is None:p.error('--output is required')
    if args.output is not None:args.output=args.output.resolve()
    if args.plan_only:
        print(json.dumps(dict(stage=args.stage,arms=args.arm,seeds=args.seeds,initialization='default',
            variants=args.variants,models=len(args.variants)*len(args.arm)*len(args.seeds),epochs=args.epochs,scheduler_epochs=64,
            estimated_hours_t4=round(len(args.arm)*len(args.seeds)*65/60,1),devices=args.devices,output=str(args.output)),indent=2))
        return
    if args.worker:worker(args)
    else:coordinator(args)


if __name__=='__main__':main()
