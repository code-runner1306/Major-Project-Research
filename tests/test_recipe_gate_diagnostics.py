import torch
from diagnose_recipe_gates import sensitivity,probe,parameter_fingerprint
from recipe_model import RecipeLM


def test_analytic_sensitivity_matches_autograd():
    raw=torch.tensor([-.2,.3],dtype=torch.float64,requires_grad=True)
    rate=torch.tensor([2.,3.],dtype=torch.float64);offset=torch.tensor([-3.,-4.],dtype=torch.float64)
    alpha=torch.exp(-rate*torch.nn.functional.softplus(raw+offset))
    grad=torch.autograd.grad(alpha.sum(),raw)[0]
    assert torch.allclose(sensitivity(raw,rate,offset),-grad)


def test_probe_preserves_parameters_and_modes():
    torch.set_num_threads(1)
    m=RecipeLM('channel-wise',128,32,16,'chunked',42)
    m.train();m.network.backbone.layers[0].eval()
    modes=[layer.training for layer in m.modules()];before=parameter_fingerprint(m)
    tokens=torch.randint(1,128,(2,32));labels=torch.full_like(tokens,-100);labels[:,15]=4
    output=probe(m,tokens,labels,'mqar',batch=1)
    assert len(output)==4 and all(row['alpha']['minimum']>=0 for row in output.values())
    assert [layer.training for layer in m.modules()]==modes
    assert parameter_fingerprint(m)==before
    assert all(not layer._forward_pre_hooks for layer in m.modules())


def test_shared_mean_intervention_preserves_mean_and_restores_function():
    from ablate_recipe_gates import shared_mean_gates
    from recipe_model import RecipeGDN
    m=RecipeLM('channel-wise',128,32,16,'chunked',42)
    layer=next(x for x in m.modules() if isinstance(x,RecipeGDN))
    x=torch.randn(2,32,16)
    before=layer.decay(x).detach();fingerprint=parameter_fingerprint(m)
    with shared_mean_gates(m):
        shared=layer.decay(x).detach()
        assert torch.allclose(shared.mean(-1),before.mean(-1))
        assert torch.allclose(shared,shared[...,:1].expand_as(shared))
    assert torch.equal(layer.decay(x).detach(),before)
    assert parameter_fingerprint(m)==fingerprint and 'decay' not in layer.__dict__
