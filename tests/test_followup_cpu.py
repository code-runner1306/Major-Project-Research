import torch
from stronger_gate_tests import intervention,static_means
from stress_recipe_checkpoints import make_mqar,make_overwrite,stale_rate,evaluate_case
from recipe_model import RecipeLM,RecipeGDN
from recipe_evaluation import evaluate


def test_shuffle_preserves_multiset_and_static_removes_input_dependence():
    torch.set_num_threads(1)
    model=RecipeLM('channel-wise',128,32,16,'chunked',42)
    layer=next(x for x in model.modules() if isinstance(x,RecipeGDN));x=torch.randn(2,32,16)
    original=layer.decay(x).detach()
    with intervention(model,'shuffle',0,90221):
        changed=layer.decay(x).detach()
        assert torch.equal(original.sort(-1).values,changed.sort(-1).values)
        assert not torch.equal(original,changed)
    with intervention(model,'static',0,means={0:original.mean((0,1))}):
        assert torch.equal(layer.decay(x),layer.decay(x+1))
    assert torch.equal(layer.decay(x).detach(),original) and 'decay' not in layer.__dict__


def test_stress_cases_and_multi_stale_metric():
    for length,keys in [(64,4),(64,8),(128,4),(128,8)]:
        case=make_mqar(4,500042,length,keys)
        assert case.audit['passed'] and (case.data.labels!=-100).sum()==4*keys
    for keys,distance,rewrites in [(4,8,1),(8,8,1),(4,24,1),(4,8,3)]:
        case=make_overwrite(4,500042,keys,distance,rewrites,100042)
        assert case.audit['passed'] and case.stale_values.shape[-1]==rewrites
        assert (case.stale_values[case.data.kinds==1]>=0).sum()==4*rewrites
    assert stale_rate(torch.tensor([10,30,40]),torch.tensor([1,1,2]),
                      torch.tensor([[10,11,12],[20,21,22],[-1,-1,-1]]))==.5


def test_baseline_stress_evaluator_matches_existing_contract():
    torch.set_num_threads(1)
    model=RecipeLM('scalar',8192,64,16,'chunked',42)
    case=make_overwrite(2,500042,4,8,1,100042)
    expected=evaluate(model,case.data,torch.device('cpu'))
    actual=evaluate_case(model,case)
    assert actual==expected and 'scored_logits' not in model.__dict__
