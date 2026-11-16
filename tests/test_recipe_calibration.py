import copy
import pytest
import torch
from recipe_calibration import calibrate, fit_offset, qualify_calibrated
from recipe_model import RecipeLM
from run_recipe_study import learning_summary


def test_offsets_only_rng_and_modes_preserved():
    torch.set_num_threads(1)
    for variant in ['scalar','channel-wise']:
        model=RecipeLM(variant,128,32,16,'chunked',42)
        model.train();model.network.backbone.layers[0].eval()
        modes=[m.training for m in model.modules()]
        before=copy.deepcopy(model.state_dict())
        tokens=torch.randint(1,128,(8,32));rng=torch.get_rng_state().clone()
        report=calibrate(model,tokens,batch=4)
        assert report['passed']
        assert torch.equal(torch.get_rng_state(),rng)
        assert [m.training for m in model.modules()]==modes
        assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items() if not name.endswith('dt_bias'))
        assert max(x['maximum_error'] for x in report['layers'])<=1e-6
        assert all('forward_pre_hooks' not in name for name in model.state_dict())
        from recipe_model import RecipeGDN
        observed=[];handles=[]
        for layer in model.modules():
            if isinstance(layer,RecipeGDN):
                handles.append(layer.register_forward_pre_hook(
                    lambda module,inputs: observed.append(module.decay(inputs[0]).detach().mean((0,1)))))
        model.eval()
        with torch.no_grad():model.network.backbone(tokens)
        for handle in handles:handle.remove()
        assert all((actual-row['target_mean_decay']).abs().max()<1e-6
                   for actual,row in zip(observed,report['layers']))
        assert qualify_calibrated(model,tokens[:2])['passed']


def test_invalid_and_unbracketed_targets():
    with pytest.raises(ValueError):fit_offset(torch.zeros(2,1,1),torch.zeros(1,1),.9)
    with pytest.raises(ValueError):fit_offset(torch.zeros(2,1,1),torch.ones(1,1),1.)
    with pytest.raises(ValueError):fit_offset(torch.zeros(2,1,1),torch.full((1,1),1e-20),.5)


def test_summary_uses_three_validations_and_fixed_area_window():
    rows=[dict(epoch=i+1,step=(i+1)*196,answer_free=dict(exact_query_accuracy=1.,first_query_accuracy=1.,
        later_query_accuracy=1.,latest_value_accuracy=None,unrelated_association_retention=None)) for i in range(16)]
    s=learning_summary(rows)
    assert s['accuracy_aulc_epochs1_16']==1.
    assert s['first_step_of_three_consecutive_95']==196
    assert s['qualification_pass']
    assert learning_summary(rows[:2])['accuracy_aulc_epochs1_16'] is None


def test_publication_includes_checkpoint_in_master_folder(tmp_path,monkeypatch):
    import json
    import os
    from contextlib import contextmanager
    import kaggle_session
    import recipe_study_artifacts as artifact
    pair=tmp_path/'pair';(pair/'scalar').mkdir(parents=True)
    (pair/'scalar'/'latest_checkpoint.pt').write_bytes(b'checkpoint-test')
    (pair/'pair_status.json').write_text(json.dumps(dict(family='overwrite',seed=42,initialization='calibrated')))
    @contextmanager
    def environment():yield dict(os.environ)
    monkeypatch.setattr(kaggle_session,'git_environment',environment)
    monkeypatch.setattr(kaggle_session,'remote_url',lambda name:'https://example.invalid/repo.git')
    pushes=[]
    def command(args,**kwargs):
        if args[1]=='push':
            files=list(kwargs['cwd'].rglob('latest_checkpoint.pt'))
            assert len(files)==1 and files[0].read_bytes()==b'checkpoint-test'
            assert 'calibrated_recipe_study' in files[0].parts
            manifest=json.loads(next(kwargs['cwd'].rglob('export_manifest.json')).read_text())
            assert manifest['checkpoints_included']
            assert manifest['files'][0]['sha256']
            pushes.append(args)
    monkeypatch.setattr(artifact.subprocess,'run',command)
    branch=artifact.publish_run(pair,'scalar','test-run','matched-decay-v1')
    assert branch.endswith('/test-run/scalar') and len(pushes)==1


def test_study_rejects_cpu_pilot_evidence(tmp_path):
    import json
    from types import SimpleNamespace
    from run_recipe_study import qualification
    (tmp_path/'pair_status.json').write_text(json.dumps(dict(family='overwrite',initialization='calibrated',
        stage='pilot',status='complete',qualification_pass=True,cpu_smoke=True)))
    args=SimpleNamespace(cpu_smoke=False,qualification_root=tmp_path)
    with pytest.raises(ValueError,match='No matching successful GPU'):
        qualification(args,'overwrite')


def test_calibration_failure_rolls_back_offsets_and_removes_hooks(monkeypatch):
    import recipe_calibration
    model=RecipeLM('scalar',128,32,16,'chunked',42)
    before=copy.deepcopy(model.state_dict());calls=[]
    original=recipe_calibration.fit_offset
    def failing(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise ValueError('Injected solver failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(recipe_calibration,'fit_offset',failing)
    with pytest.raises(ValueError,match='Injected'):
        calibrate(model,torch.randint(1,128,(2,32)))
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in before.items())
    assert all(not module._forward_pre_hooks for module in model.modules())
