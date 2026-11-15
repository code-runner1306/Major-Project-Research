"""Real-task timing/development pair for the pinned MQAR recipe adaptation.

Never evaluates confirmation data or starts the final factorial study. A
development budget requires matching real-task GPU timing evidence first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from artifacts import atomic_write_json
from recipe_data import audit_mqar,build_data,split_overlap
from recipe_evaluation import evaluate
from recipe_model import FLA_REVISION,RecipeLM


def source_identity():
    names=('recipe_model.py','recipe_data.py','recipe_evaluation.py',
           'experimental_chunked_recurrence.py','third_party/zoology_iclr24/model.py',
           'third_party/zoology_iclr24/compat.py','third_party/zoology_iclr24/attention.py','run_published_recipe.py')
    return {name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names}


def settings(args):
    return {'family':args.family,'backend':args.backend,'seed':args.seed,
        'vocabulary':128 if args.cpu_smoke else 8192,'length':32 if args.cpu_smoke else 64,
        'width':16 if args.cpu_smoke else 128,'train_examples':32 if args.cpu_smoke else 100000,
        'validation_examples':16 if args.cpu_smoke else args.validation_examples,
        'microbatch':args.microbatch,'effective_batch':args.effective_batch,
        'validation_interval':'each epoch and at the end of the timing window',
        'epochs':args.epochs,'learning_rate':args.learning_rate,'gradient_clipping':None,
        'optimizer':'AdamW','weight_decay':.1,'timescale_parameter_weight_decay':0.,
        'precision':'float32','torch_version':torch.__version__,
        'scheduler':'CosineAnnealingLR stepped per complete epoch; no warmup',
        'source_hashes':source_identity(),'fla_reference_revision':FLA_REVISION,
        'adaptations':['GDN mixer extension','V4 overwrite extension','stable paired GDN initialization',
            'microbatch gradient accumulation','fixed budget instead of per-model early stopping',
            'validation512 by default instead of published3000','separate future confirmation set']}


def validate_gpu_candidate(args):
    if args.cpu_smoke or args.backend=='reference':
        return
    if args.qualification is None:
        raise ValueError('Chunked GPU execution requires --qualification from the implementation pilot')
    status=json.loads((args.qualification/'pilot_status.json').read_text())
    numerical=json.loads((args.qualification/'numerical_qualification.json').read_text())
    if len(status.get('gpus',[]))<2 or not numerical.get('passed') or not numerical.get('model_trajectory',{}).get('passed'):
        raise ValueError('Candidate qualification must contain GPU numerical and model trajectory passes')
    if status['gpus'] != [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]:
        raise ValueError('GPU hardware differs from implementation qualification')
    if status.get('torch') != torch.__version__:
        raise ValueError('PyTorch version differs from implementation qualification')
    if status.get('recipe_model_sha256')!=source_identity()['recipe_model.py'] or status.get('candidate_source_sha256')!=source_identity()['experimental_chunked_recurrence.py']:
        raise ValueError('Candidate qualification source hashes do not match current code')
    expected_backbone={name:value for name,value in source_identity().items() if name.startswith('third_party/')}
    if status.get('recipe_backbone_sha256')!=expected_backbone:
        raise ValueError('Published backbone changed after implementation qualification')
    phases=[row for row in status['phases'] if row['backend']=='chunked' and row['workers']==2]
    if not phases or any(row['status']!='complete' for row in phases[0]['workers_summary']):
        raise ValueError('Two-worker chunked GPU timing did not complete')


def timing_budget_guard(args):
    if args.mode=='timing':
        return None
    if args.timing_evidence is None:
        raise ValueError('Development requires --timing-evidence from a real-task timing pair')
    session=json.loads((args.timing_evidence/'session_status.json').read_text())
    if session.get('status')!='complete' or session.get('mode')!='timing':
        raise ValueError('Real-task timing pair must have completed')
    current=settings(args)
    estimates={}
    for variant in ('scalar','channel-wise'):
        summary=json.loads((args.timing_evidence/variant/'summary.json').read_text())
        if summary.get('status')!='complete' or summary.get('mode')!='timing':
            raise ValueError('Timing worker must have completed in timing mode')
        old=summary['settings']
        for key in ('family','backend','vocabulary','length','width','train_examples',
                    'validation_examples','microbatch','effective_batch','source_hashes','torch_version','precision'):
            if old[key]!=current[key]:
                raise ValueError(f'Real-task timing mismatch: {variant}/{key}')
        if not args.cpu_smoke and (summary.get('gpu') is None or summary['accepted_optimizer_steps']<50):
            raise ValueError('At least50 accepted GPU optimizer steps are required for timing')
        if not args.cpu_smoke and summary['gpu']!=torch.cuda.get_device_name(0 if variant=='scalar' else 1):
            raise ValueError('Real-task timing hardware differs')
        durations=sorted(summary['optimizer_step_seconds'])
        if (not durations or len(durations)!=summary['accepted_optimizer_steps']
                or any(not math.isfinite(value) or value<=0 for value in durations)):
            raise ValueError('Timing samples must be finite, positive and match accepted steps')
        if any(not math.isfinite(summary[key]) or summary[key]<0
               for key in ('setup_seconds','validation_seconds','checkpoint_seconds')):
            raise ValueError('Timing overhead must be finite and nonnegative')
        conservative=max(statistics.mean(durations)*1.2,durations[math.ceil(.95*len(durations))-1])
        steps=math.ceil(current['train_examples']/current['effective_batch'])*args.epochs
        # Include both evaluation histories, worst-case checkpoint per epoch,
        # one future confirmation-size pass, doubled setup and 15m reserve.
        estimate=steps*conservative+(args.epochs+1)*summary['validation_seconds']*1.2
        estimate+=args.epochs*summary['checkpoint_seconds']+2*summary['setup_seconds']+900
        estimates[variant]=estimate
    if max(estimates.values())>args.max_seconds:
        raise ValueError(f'Requested fixed budget exceeds conservative session limit: {estimates}')
    return estimates


def worker(args):
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    cfg=settings(args)
    atomic_write_json(args.output/'settings.json',cfg)
    train=build_data(args.family,cfg['train_examples'],args.seed+100000,'train',cfg['vocabulary'],cfg['length'])
    dev=build_data(args.family,cfg['validation_examples'],args.seed+200000,'validation',cfg['vocabulary'],cfg['length'])
    atomic_write_json(args.output/'data_manifests.json',{'train':train.identity,'development':dev.identity})
    if args.family=='mqar':
        audit=audit_mqar(dev.tokens,dev.labels)
    else:
        from config import RetrievalConfig
        from retrieval import RetrievalDataset,evaluate_retrieval_baselines
        config=RetrievalConfig(vocabulary_size=cfg['vocabulary'],sequence_lengths=(cfg['length']+1,),
            key_counts=(4,),query_distances=(8,),overwrite_counts=(1,),generator_version='mqar-overwrite-v4')
        sample=RetrievalDataset(task='overwrite',examples=max(128,cfg['validation_examples']),config=config,
            seed=args.seed+200000,split='validation')
        frequency=RetrievalDataset(task='overwrite',examples=256,config=config,seed=args.seed+100000,split='train')
        audit=evaluate_retrieval_baselines(sample,frequency_dataset=frequency)
        audit['passed']=(audit['status']=='complete' and audit['oracle_audit']['pass']
            and audit['counterfactual_query_key_audit']['pass'] and audit['shortcut_criterion']['all_cells_pass']
            and audit['completeness']['incomplete_checks']==0)
    audit['train_development_content_overlap']=split_overlap(train,dev)
    audit['passed']=audit['passed'] and audit['train_development_content_overlap']==0
    atomic_write_json(args.output/'data_audit.json',audit)
    if not audit['passed']:
        raise RuntimeError('Real-task data audit failed; no training')
    model=RecipeLM(args.variant,cfg['vocabulary'],cfg['length'],cfg['width'],args.backend,args.seed).to(device)
    optimizer=model.optimizer(args.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=0.)
    torch.manual_seed(args.seed+1000)
    setup=time.monotonic()-start
    curves=[]
    durations=[]
    validation_seconds=checkpoint_seconds=0.
    accepted=0
    last_loss=None
    model.train()
    for epoch in range(args.epochs):
        for offset in range(0,len(train.tokens),args.effective_batch):
            if time.monotonic()-start>args.max_seconds-600:
                raise TimeoutError('Time reserve reached; worker is incomplete')
            tick=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            actual=min(args.effective_batch,len(train.tokens)-offset)
            loss_sum=0.
            for micro in range(offset,offset+actual,args.microbatch):
                stop=min(micro+args.microbatch,offset+actual)
                tokens=train.tokens[micro:stop].to(device)
                labels=train.labels[micro:stop].to(device)
                loss=model.loss(tokens,labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                weighted=loss*((stop-micro)/actual)
                weighted.backward()
                loss_sum+=weighted.item()
            norm=torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None],foreach=True)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
            if device.type=='cuda': torch.cuda.synchronize(device)
            durations.append(time.monotonic()-tick)
            accepted+=1
            last_loss=loss_sum
            if accepted%50==0:
                print(f'{args.family}/{args.variant}: {accepted} accepted steps',flush=True)
            if args.mode=='timing' and accepted>=args.steps:
                break
        tick=time.monotonic()
        answer_free=evaluate(model,dev,device,batch=args.microbatch)
        teacher=(evaluate(model,dev,device,batch=args.microbatch,history='teacher_forced')
                 if args.family=='overwrite' else dict(answer_free,history_conditioning='teacher_forced'))
        elapsed=time.monotonic()-tick
        validation_seconds+=elapsed
        curves.append({'epoch':epoch+1,'step':accepted,'learning_rate':optimizer.param_groups[0]['lr'],
            'train_loss':last_loss,'answer_free':answer_free,'teacher_forced':teacher,'validation_seconds':elapsed})
        atomic_write_json(args.output/'development_curve.json',curves)
        tick=time.monotonic()
        complete_epoch=offset+actual==len(train.tokens)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'completed_steps':accepted,
            'completed_epochs':epoch+int(complete_epoch),'settings':cfg},args.output/'latest_checkpoint.pt')
        checkpoint_seconds+=time.monotonic()-tick
        print(json.dumps(curves[-1]),flush=True)
        if args.mode=='timing' and accepted>=args.steps: break
        if complete_epoch: scheduler.step()
    expected=args.steps if args.mode=='timing' else math.ceil(len(train.tokens)/args.effective_batch)*args.epochs
    if accepted!=expected:
        raise RuntimeError(f'Accepted {accepted} optimizer steps, expected {expected}')
    atomic_write_json(args.output/'summary.json',{'status':'complete','mode':args.mode,'variant':args.variant,
        'settings':cfg,'accepted_optimizer_steps':accepted,'optimizer_step_seconds':durations,
        'setup_seconds':setup,'validation_seconds':validation_seconds,'checkpoint_seconds':checkpoint_seconds,
        'wall_seconds':time.monotonic()-start,'last_development':curves[-1],
        'parameters':sum(p.numel() for p in model.parameters()),
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'interpretation':('Timing window is not learnability qualification. Confirmation has not been evaluated.'
            if args.mode=='timing' else 'Development evidence only. Confirmation has not been evaluated.')})


def coordinator(args):
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    status={'status':'running','mode':args.mode,'family':args.family,'jobs':{},
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    processes=[]
    logs=[]
    code=0
    try:
        validate_gpu_candidate(args)
        status['conservative_budget_seconds']=timing_budget_guard(args)
        if not args.cpu_smoke and torch.cuda.device_count()<2:
            raise RuntimeError('Two CUDA GPUs required')
        for gpu,variant in enumerate(('scalar','channel-wise')):
            log=(args.output/f'{variant}.log').open('w')
            logs.append(log)
            command=[sys.executable,__file__,'--worker','--variant',variant,'--family',args.family,
                '--mode',args.mode,'--backend',args.backend,'--epochs',str(args.epochs),'--steps',str(args.steps),
                '--microbatch',str(args.microbatch),'--effective-batch',str(args.effective_batch),
                '--validation-examples',str(args.validation_examples),'--seed',str(args.seed),
                '--learning-rate',str(args.learning_rate),'--max-seconds',str(args.max_seconds),
                '--output',str(args.output/variant)]
            if args.cpu_smoke: command.append('--cpu-smoke')
            processes.append(subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):
                raise RuntimeError('Recipe worker failed; stopping its paired worker')
            if time.monotonic()-start>args.max_seconds-300:
                raise TimeoutError('Coordinator reserve reached')
            time.sleep(1)
        status['jobs']={v:p.returncode for v,p in zip(('scalar','channel-wise'),processes)}
        if any(p.returncode for p in processes):
            raise RuntimeError('A recipe worker failed; inspect logs')
        left=json.loads((args.output/'scalar'/'data_manifests.json').read_text())
        right=json.loads((args.output/'channel-wise'/'data_manifests.json').read_text())
        if left!=right:
            raise RuntimeError('Paired dataset identities differ')
        status['status']='complete'
    except (Exception,KeyboardInterrupt) as error:
        status.update(status='failed_or_incomplete',error=str(error))
        code=1
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill();p.wait()
        for log in logs: log.close()
        status['wall_seconds']=time.monotonic()-start
        atomic_write_json(args.output/'session_status.json',status)
        print(json.dumps(status),flush=True)
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--family',choices=('mqar','overwrite'),required=True)
    p.add_argument('--variant',choices=('scalar','channel-wise'))
    p.add_argument('--worker',action='store_true')
    p.add_argument('--mode',choices=('timing','development'),default='timing')
    p.add_argument('--backend',choices=('reference','chunked'),default='reference')
    p.add_argument('--qualification',type=Path)
    p.add_argument('--timing-evidence',type=Path)
    p.add_argument('--epochs',type=int,default=64)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--learning-rate',type=float,default=10**(-8/3))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--microbatch',type=int,default=32)
    p.add_argument('--effective-batch',type=int,default=512)
    p.add_argument('--validation-examples',type=int,default=512)
    p.add_argument('--max-seconds',type=int,default=39600)
    p.add_argument('--cpu-smoke',action='store_true')
    args=p.parse_args()
    if min(args.epochs,args.steps,args.microbatch,args.effective_batch,args.validation_examples)<1 or args.epochs>64:
        p.error('Invalid training dimensions or epoch budget')
    if args.microbatch>args.effective_batch or not math.isfinite(args.learning_rate) or not 0<args.learning_rate or not 900<args.max_seconds<=39600:
        p.error('Invalid batch, learning rate or session ceiling')
    maximum=math.ceil((32 if args.cpu_smoke else 100000)/args.effective_batch)*args.epochs
    if args.mode=='timing' and args.steps>maximum:
        p.error('Timing steps exceed the configured epoch capacity')
    if args.worker and args.variant is None: p.error('Worker requires variant')
    args.output=args.output.resolve()
    if args.worker: worker(args); return 0
    return coordinator(args)


if __name__=='__main__': raise SystemExit(main())
