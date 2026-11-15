import copy
import json
from types import SimpleNamespace

import pytest

from run_published_recipe import settings,timing_budget_guard,validate_gpu_candidate


def args_for(root):
    return SimpleNamespace(mode='development',timing_evidence=root,family='mqar',backend='reference',
        seed=42,cpu_smoke=True,microbatch=2,effective_batch=8,validation_examples=512,
        epochs=2,learning_rate=.0021544346900318843,max_seconds=39600)


def timing_files(args,seconds=1.):
    root=args.timing_evidence
    root.mkdir()
    (root/'session_status.json').write_text(json.dumps({'status':'complete','mode':'timing'}))
    for variant in ('scalar','channel-wise'):
        path=root/variant
        path.mkdir()
        (path/'summary.json').write_text(json.dumps({'status':'complete','mode':'timing',
            'settings':settings(args),'accepted_optimizer_steps':2,
            'optimizer_step_seconds':[seconds,seconds],'setup_seconds':3.,
            'validation_seconds':1.,'checkpoint_seconds':.5,'gpu':None}))


def test_real_task_budget_includes_overhead_and_rejects_over_limit(tmp_path):
    args=args_for(tmp_path/'timing')
    timing_files(args)
    estimates=timing_budget_guard(args)
    # 8 steps at conservative1.2s,3 evals at1.2s,2 checkpoints,
    # doubled3s setup and900s reserve.
    assert estimates['scalar']==pytest.approx(920.2)
    args.epochs=64
    for variant in ('scalar','channel-wise'):
        path=args.timing_evidence/variant/'summary.json'
        row=json.loads(path.read_text())
        row['optimizer_step_seconds']=[1000.,1000.]
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='exceeds'):
        timing_budget_guard(args)


@pytest.mark.parametrize('mutation',['nan','source','batch'])
def test_invalid_or_incompatible_timing_cannot_authorize_development(tmp_path,mutation):
    args=args_for(tmp_path/'timing')
    timing_files(args)
    path=args.timing_evidence/'channel-wise'/'summary.json'
    row=json.loads(path.read_text())
    if mutation=='nan': row['optimizer_step_seconds'][0]=float('nan')
    if mutation=='source': row['settings']['source_hashes']['recipe_model.py']='different'
    if mutation=='batch': row['settings']['effective_batch']=16
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        timing_budget_guard(args)


def test_cpu_implementation_pass_does_not_authorize_gpu_candidate(tmp_path):
    (tmp_path/'pilot_status.json').write_text(json.dumps({'gpus':[]}))
    (tmp_path/'numerical_qualification.json').write_text(json.dumps({'passed':True,'model_trajectory':{'passed':True}}))
    args=SimpleNamespace(cpu_smoke=False,backend='chunked',qualification=tmp_path)
    with pytest.raises(ValueError,match='GPU numerical'):
        validate_gpu_candidate(args)
