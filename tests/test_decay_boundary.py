import torch
from evaluate_decay_boundary import (audit_overwrite,cases_for,paired_case,pair_metrics,score,fit_constants,qualify_forward,partition_checkpoints)
from recipe_evaluation import evaluate
from recipe_model import RecipeLM
from stronger_gate_tests import intervention


def test_all_discovery_cells_and_counterfactual_oracles():
    torch.set_num_threads(1)
    for family in ('mqar','overwrite'):
        cases=cases_for(family,42,3)
        assert len(cases)==(6 if family=='mqar' else 9)
        assert all(c.audit['passed'] for c in cases.values())
        if family=='overwrite':
            base=cases['rewrites3'];mask=base.data.labels!=-100
            gold=base.data.labels[mask]
            for kind in ('obsolete_prefix','latest_value','rename_key'):
                changed=cases['pair_'+kind]
                assert audit_overwrite(changed)
                metrics=pair_metrics(base,changed,gold,changed.data.labels[mask],kind)
                assert metrics['latest_joint_correct']==metrics['unrelated_joint_correct']==1.
                assert metrics['latest_changed_fraction']==(1. if kind=='latest_value' else 0.)
                assert metrics['unrelated_changed_fraction']==0.


def test_scoring_preserves_existing_contract_and_counts_any_stale_value():
    torch.set_num_threads(1)
    case=cases_for('overwrite',42,2)['baseline']
    model=RecipeLM('scalar',8192,64,16,'chunked',42)
    expected=evaluate(model,case.data,torch.device('cpu'),batch=1)
    actual,predictions=score(model,case,torch.device('cpu'),1)
    assert actual==expected and len(predictions)==4 and 'scored_logits' not in model.__dict__
    case=cases_for('overwrite',42,2)['rewrites3']
    prior=case.stale_values[case.data.kinds==1]
    assert prior.shape==(2,3)


def test_constants_are_training_only_and_intervention_restores_parameters():
    torch.set_num_threads(1)
    case=cases_for('overwrite',42,2)['baseline']
    model=RecipeLM('channel-wise',8192,64,16,'chunked',42)
    before={k:v.clone() for k,v in model.state_dict().items()}
    means=fit_constants(model,case.data,torch.device('cpu'),2,1)
    changed_labels=case.data.labels.clone();case.data.labels.fill_(-100)
    means2=fit_constants(model,case.data,torch.device('cpu'),2,2)
    case.data.labels=changed_labels
    assert all(torch.allclose(means[k],means2[k],atol=1e-6) for k in means)
    for mode in ('static','mean'):
        with intervention(model,mode,means=means):
            assert qualify_forward(model,case,torch.device('cpu'))['passed']
            assert model.training
            score(model,case,torch.device('cpu'),1)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all('decay' not in m.__dict__ for m in model.modules())


def test_agreement_is_not_counted_as_correct_updating():
    base=cases_for('overwrite',42,2)['rewrites3'];changed=paired_case(base,'latest_value',90042)
    wrong=torch.zeros(int((base.data.labels!=-100).sum()),dtype=torch.long)
    metrics=pair_metrics(base,changed,wrong,wrong,'latest_value')
    assert metrics['latest_joint_correct']==0.
    assert metrics['latest_changed_fraction']==0.
    assert metrics['latest_correct_after_given_correct_before'] is None


def test_worker_partition_has_no_duplicates_and_keeps_paired_models_together():
    selected={(f,s,i,v):None for f in ('mqar','overwrite') for s in (42,43,44)
              for i in ('default','calibrated','joint') for v in ('scalar','channel-wise')}
    groups=partition_checkpoints(selected,2)
    assert set(groups[0]).isdisjoint(groups[1])
    assert set(groups[0])|set(groups[1])==set(selected)
    assert partition_checkpoints(dict(reversed(list(selected.items()))),2)==groups
    for group in groups:
        for key in group:
            assert (*key[:3],'scalar') in group and (*key[:3],'channel-wise') in group
    cost=lambda keys:sum((1 if k[3]=='scalar' else 3)*(6 if k[0]=='mqar' else 9) for k in keys)
    assert abs(cost(groups[0])-cost(groups[1]))<=36
