"""Bounded GPU qualification and concurrency pilot, not a learnability result."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

from artifacts import atomic_write_json
from experimental_chunked_recurrence import chunked_recurrence
from recurrence import RecurrenceOptions
from recipe_model import RecipeLM, serial_rectangular


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def numerical_qualification(device, chunk_size):
    """Resident real-shape FP32 output/state/gradient comparisons, both gates."""
    torch.manual_seed(123)
    rows = []
    options = RecurrenceOptions(chunk_size=chunk_size, state_accumulator_dtype='float32')
    for variant, gate_width in (('scalar', 1), ('channel-wise', 128)):
        for zero_gate in (False, True):
            q, k = [F.normalize(torch.randn(2, 64, 1, 128, device=device), dim=-1) for _ in range(2)]
            v = torch.randn(2,64,1,256,device=device)
            alpha = torch.rand(2, 64, 1, gate_width, device=device)
            if zero_gate:
                alpha[:, 7] = 0
            beta = torch.rand(2, 64, 1, 1, device=device)
            initial = torch.randn(2, 1, 256, 128, device=device)
            inputs = [x.requires_grad_() for x in (q, k, v, alpha, beta, initial)]
            ref, ref_state = serial_rectangular(*inputs)
            actual, state = chunked_recurrence(*inputs, options=options)
            weights, state_weights = torch.randn_like(ref), torch.randn_like(ref_state)
            reference_grads = torch.autograd.grad((ref * weights).sum() + (ref_state * state_weights).sum(), inputs)
            grads = torch.autograd.grad((actual * weights).sum() + (state * state_weights).sum(), inputs)
            errors = {}
            for name, result, expected in [('output', actual, ref), ('state', state, ref_state),
                    *[(name, result, expected) for name, result, expected in zip(
                        ('q_grad','k_grad','v_grad','alpha_grad','beta_grad','initial_state_grad'), grads, reference_grads)]]:
                absolute = (result - expected).abs().max().item()
                relative = ((result - expected).norm() / expected.norm().clamp_min(1e-12)).item()
                errors[name] = {'max_absolute': absolute, 'normalized_l2': relative}
                if not torch.isfinite(result).all() or absolute > 2e-4 or relative > 2e-4:
                    raise RuntimeError(f'Chunked qualification failed: {variant}/{name}: {errors[name]}')
            rows.append({'variant': variant, 'zero_gate': zero_gate, 'errors': errors})
    return {'passed': True, 'precision': 'float32', 'chunk_size': chunk_size, 'cases': rows}


def make_model(variant, backend, chunk_size, device):
    model = RecipeLM(variant,vocabulary=8192,length=64,width=128,backend=backend,seed=42)
    for layer in model.network.backbone.layers:
        layer.sequence_mixer.chunk_size=chunk_size
    return model.to(device)


def model_qualification(device, chunk_size):
    rows = []
    for variant in ('scalar','channel-wise'):
        reference = make_model(variant, 'reference', chunk_size, device)
        candidate = copy.deepcopy(reference)
        for layer in candidate.network.backbone.layers:
            layer.sequence_mixer.backend='chunked'
            layer.sequence_mixer.chunk_size=chunk_size
        optimizers = [m.optimizer(1e-3) for m in (reference,candidate)]
        torch.manual_seed(991)
        max_error = 0.
        max_relative = 0.
        for step_index in range(5):
            tokens = torch.randint(1,8192,(2,64),device=device)
            labels = torch.full_like(tokens,-100)
            labels[:,[15,23,39,55]]=torch.randint(4096,8192,(2,4),device=device)
            for model,optimizer in zip((reference,candidate),optimizers):
                torch.manual_seed(12000+step_index)  # Identical published dropout masks.
                optimizer.zero_grad(set_to_none=True)
                if model is candidate:
                    loss=model.loss(tokens,labels)
                else:
                    _,loss=model(tokens,labels)
                loss.backward()
                norm=torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None],foreach=True)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite qualification training trajectory')
                optimizer.step()
            for actual,expected in zip(candidate.parameters(),reference.parameters()):
                max_error=max(max_error,(actual-expected).abs().max().item())
                relative=((actual-expected).norm()/expected.norm().clamp_min(1e-12)).item()
                max_relative=max(max_relative,relative)
                # Adam magnifies round-off for nearly zero gradients. Bound both
                # worst coefficient drift and whole-tensor drift; do not demand
                # bitwise agreement between different FP32 reduction orders.
                if max_error>1e-4 or max_relative>5e-4:
                    raise RuntimeError(f'Optimizer trajectory mismatch: {variant}: {max_error=}, {max_relative=}')
        rows.append({'variant':variant,'optimizer_steps':5,'max_parameter_absolute_error':max_error,
                     'max_parameter_normalized_l2_error':max_relative})
    return {'passed':True,'comparison':'reference/full logits vs chunked/scored-token loss','cases':rows}


def worker(args):
    import psutil
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    started = time.monotonic()
    model = make_model(args.variant, args.backend, args.chunk_size, device)
    optimizer = model.optimizer(1e-3)
    # Timing uses generated resident tensors. These scores must never qualify a task.
    generator = torch.Generator().manual_seed(771)
    tokens = torch.randint(1, 8192, (args.batch, 64), generator=generator).to(device)
    labels = torch.full_like(tokens, -100)
    labels[:, [15, 23, 39, 55]] = torch.randint(4096, 8192, (args.batch, 4), generator=generator).to(device)
    def step():
        optimizer.zero_grad(set_to_none=True)
        loss_value=0.
        for _ in range(args.accumulation):
            loss=model.loss(tokens,labels,scored_only=args.loss_implementation=='scored')
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pilot loss')
            (loss/args.accumulation).backward()
            loss_value+=loss.item()/args.accumulation
        # Pinned trainer does not clip. Check gradients without changing them.
        norm=torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None],foreach=True)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite pilot gradient')
        optimizer.step()
        return loss_value
    setup = time.monotonic() - started
    warm_start = time.monotonic()
    for _ in range(args.warmup):
        step()
    synchronize(device)
    warmup_seconds = time.monotonic() - warm_start
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    process = psutil.Process()
    cpu_start = process.cpu_times()
    for _ in range(args.steps):
        synchronize(device)
        tick = time.monotonic()
        loss = step()
        synchronize(device)
        durations.append(time.monotonic() - tick)
    cpu_end = process.cpu_times()
    eval_start = time.monotonic()
    model.eval()
    with torch.no_grad():
        for offset in range(0,512,args.batch):
            size=min(args.batch,512-offset)
            model(tokens[:size], labels[:size])
    synchronize(device)
    eval_seconds = time.monotonic() - eval_start
    checkpoint_start = time.monotonic()
    checkpoint = args.output.with_suffix('.pt')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
    checkpoint_seconds = time.monotonic() - checkpoint_start
    summary = {'status': 'complete', 'backend': args.backend, 'variant': args.variant,
        'loss_implementation':args.loss_implementation,
        'measured_optimizer_steps': args.steps, 'batch': args.batch, 'sequence_length': 64,
        'gradient_accumulation':args.accumulation,'effective_batch':args.batch*args.accumulation,
        'model_recipe':'pinned-zoology-backbone-with-FLA-based-GDN-extension',
        'parameters': sum(p.numel() for p in model.parameters()), 'precision': 'float32',
        'setup_seconds': setup, 'warmup_seconds': warmup_seconds,
        'optimizer_step_seconds': durations, 'mean_step_seconds': statistics.mean(durations),
        'evaluation_512_examples_seconds': eval_seconds, 'checkpoint_seconds': checkpoint_seconds,
        'checkpoint_bytes': checkpoint.stat().st_size, 'last_loss': loss,
        'process_cpu_seconds_during_measurement': cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system,
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        'process_rss_bytes': process.memory_info().rss, 'total_seconds': time.monotonic() - started,
        'warning': 'Synthetic resident tensors measure implementation throughput, not retrieval learning.'}
    atomic_write_json(args.output, summary)


def coordinator(args):
    import psutil
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    device = torch.device('cpu' if args.cpu_smoke else 'cuda:0')
    if not args.cpu_smoke and torch.cuda.device_count() < 2:
        raise RuntimeError('Two CUDA GPUs required')
    metadata = {'status': 'running', 'source_revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        'candidate_source_sha256': hashlib.sha256(Path('experimental_chunked_recurrence.py').read_bytes()).hexdigest(),
        'recipe_model_sha256':hashlib.sha256(Path('recipe_model.py').read_bytes()).hexdigest(),
        'recipe_backbone_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            'third_party/zoology_iclr24/model.py','third_party/zoology_iclr24/compat.py',
            'third_party/zoology_iclr24/attention.py')},
        'torch': torch.__version__, 'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'purpose': 'Numerical and timing qualification only; no learned accuracy claim', 'phases': []}
    atomic_write_json(args.output/'pilot_status.json', metadata)
    try:
        numerical = numerical_qualification(device, args.chunk_size)
        numerical['model_trajectory'] = model_qualification(device,args.chunk_size)
        atomic_write_json(args.output/'numerical_qualification.json', numerical)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for trial_backend,trial_loss in (('reference','full'),('reference','scored'),('chunked','scored')):
            backend,loss_implementation=trial_backend,trial_loss
            for count in args.workers:
                # Reference oversubscription is unnecessary if the candidate loses.
                if trial_backend == 'reference' and count != 2:
                    continue
                phase_start = time.monotonic()
                processes, logs, paths = [], [], []
                cpu_samples = []
                try:
                    for index in range(count):
                        name = f'{backend}_{loss_implementation}_workers{count}_{index}'
                        path = args.output/f'{name}.json'
                        paths.append(path)
                        log = (args.output/f'{name}.log').open('w')
                        logs.append(log)
                        command = [sys.executable,__file__,'--worker','--backend',backend,
                            '--loss-implementation',loss_implementation,
                            '--variant','scalar' if index % 2 == 0 else 'channel-wise',
                            '--batch',str(args.batch),'--steps',str(args.steps),'--warmup',str(args.warmup),
                            '--accumulation',str(args.accumulation),
                            '--chunk-size',str(args.chunk_size),'--output',str(path)]
                        if args.cpu_smoke:
                            command.append('--cpu-smoke')
                        processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index % 2),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                            start_new_session=False))
                    psutil.cpu_percent()
                    while any(p.poll() is None for p in processes):
                        if any(p.poll() not in (None,0) for p in processes):
                            raise RuntimeError(f'Worker failed in {backend}/{count}; stopping the wave')
                        if time.monotonic()-start > args.max_seconds:
                            raise TimeoutError('Pilot session limit reached')
                        cpu_samples.append(psutil.cpu_percent())
                        time.sleep(1)
                    if any(p.returncode for p in processes):
                        raise RuntimeError(f'Worker failed in {backend}/{count}; inspect logs')
                finally:
                    for p in processes:
                        if p.poll() is None:
                            p.terminate()
                    for p in processes:
                        try:
                            p.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            p.kill()
                            p.wait()
                    for log in logs:
                        log.close()
                summaries = [json.loads(p.read_text()) for p in paths]
                elapsed = time.monotonic() - phase_start
                row = {'backend':backend,'loss_implementation':loss_implementation,'workers':count,'phase_wall_seconds':elapsed,
                    'end_to_end_measured_steps_per_second': count*args.steps/elapsed,
                    'cpu_percent_mean': statistics.mean(cpu_samples) if cpu_samples else None,
                    'workers_summary':summaries}
                metadata['phases'].append(row)
                atomic_write_json(args.output/'pilot_status.json', metadata)
                print(json.dumps({k:v for k,v in row.items() if k!='workers_summary'}),flush=True)
                # Only run 3/4 workers when two-worker peak memory leaves room for
                # a second worker per GPU, with 25% headroom. Fresh GPU0 context
                # from numerical tests is released before these waves below.
                if trial_backend == 'chunked' and count == 2 and not args.cpu_smoke:
                    reference_phase = metadata['phases'][0]
                    def pair_time(phase):
                        return max(s['mean_step_seconds'] for s in phase['workers_summary'])
                    best=min(metadata['phases'],key=pair_time)
                    metadata['selected_two_worker_candidate']={
                        'backend':best['backend'],'loss_implementation':best['loss_implementation'],
                        'paired_mean_step_seconds':pair_time(best)}
                    if pair_time(best) >= pair_time(reference_phase):
                        metadata['oversubscription_skipped'] = 'Neither optimization improved two-worker optimizer throughput'
                        break
                    if best['cpu_percent_mean'] is not None and best['cpu_percent_mean'] > 85:
                        metadata['oversubscription_skipped'] = 'Two-worker pilot already saturated CPU capacity'
                        break
                    total_memory = min(torch.cuda.get_device_properties(i).total_memory for i in (0,1))
                    peak = max(s['peak_gpu_reserved_bytes'] for s in best['workers_summary'])
                    if peak*2 > total_memory*.75:
                        metadata['oversubscription_skipped'] = 'Two workers per GPU would exceed memory headroom'
                        break
                    # Benchmark whichever combination actually won, rather
                    # than presuming the new recurrence is the faster one.
                    backend,loss_implementation=best['backend'],best['loss_implementation']
                if count>=3 and row['cpu_percent_mean'] is not None and row['cpu_percent_mean']>90:
                    metadata['oversubscription_skipped']='Measured oversubscription saturated CPU capacity'
                    break
        metadata['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status='failed_or_incomplete',error=str(error))
        raise
    finally:
        metadata['wall_seconds'] = time.monotonic()-start
        atomic_write_json(args.output/'pilot_status.json',metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--backend',choices=('reference','chunked'),default='reference')
    parser.add_argument('--variant',choices=('scalar','channel-wise'),default='scalar')
    parser.add_argument('--loss-implementation',choices=('full','scored'),default='full')
    parser.add_argument('--workers',type=int,nargs='+',default=[2,3,4])
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--batch',type=int,default=32)
    parser.add_argument('--accumulation',type=int,default=16)
    parser.add_argument('--chunk-size',type=int,default=16)
    parser.add_argument('--max-seconds',type=int,default=5400)
    parser.add_argument('--cpu-smoke',action='store_true')
    args=parser.parse_args()
    if min(args.steps,args.batch,args.accumulation,args.chunk_size,args.max_seconds)<1 or args.warmup<0:
        parser.error('Budgets must be positive')
    if not args.workers or args.workers[0]!=2 or any(x not in (2,3,4) for x in args.workers) or args.workers!=sorted(set(args.workers)):
        parser.error('Worker counts must start at2 and increase uniquely through3/4')
    args.output=args.output.resolve()
    if args.worker: worker(args)
    else: coordinator(args)


if __name__=='__main__': main()
