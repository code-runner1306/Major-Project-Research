import copy
import math

import pytest
import torch
from torch.nn import functional as F
from joint_recipe_calibration import calibrate_joint,fit_joint,SENSITIVITIES
from recipe_calibration import TARGETS,qualify_calibrated
from recipe_model import RecipeLM,RecipeGDN


def test_joint_realized_targets_and_unchanged_backbone():
    torch.set_num_threads(1)
    for variant in ('scalar','channel-wise'):
        model=RecipeLM(variant,128,32,16,'chunked',42)
        model.train();model.network.backbone.layers[0].eval()
        modes=[m.training for m in model.modules()]
        before=copy.deepcopy(model.state_dict())
        tokens=torch.randint(1,128,(4,32));rng=torch.get_rng_state().clone()
        report=calibrate_joint(model,tokens,batch=2)
        assert report['passed'] and torch.equal(rng,torch.get_rng_state())
        assert modes==[m.training for m in model.modules()]
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()
                   if not k.endswith(('dt_bias','A_log')))
        observed=[];handles=[]
        def capture(module,inputs):
            raw=module.a_proj(inputs[0]).reshape(-1,module.heads,module.gate_width).double()
            shifted=raw+module.dt_bias.double()
            observed.append((shifted.sigmoid().mean(0),
                -module.A_log.double().exp()*F.softplus(shifted).mean(0)))
        for layer in model.modules():
            if isinstance(layer,RecipeGDN):handles.append(layer.register_forward_pre_hook(capture))
        model.eval()
        with torch.no_grad():model.network.backbone(tokens)
        for handle in handles:handle.remove()
        for (s,d),target,half in zip(observed,SENSITIVITIES,TARGETS):
            assert (s-target).abs().max()<1e-6
            assert (d+math.log(2)/half).abs().max()<1e-6
        assert qualify_calibrated(model,tokens[:2])['passed']


def test_joint_failure_rolls_back_all_memory_parameters(monkeypatch):
    import joint_recipe_calibration as joint
    model=RecipeLM('channel-wise',128,32,16,'chunked',42)
    before=copy.deepcopy(model.state_dict());original=joint.fit_joint;calls=[]
    def fail(*args):
        calls.append(1)
        if len(calls)==2:raise ValueError('injected')
        return original(*args)
    monkeypatch.setattr(joint,'fit_joint',fail)
    with pytest.raises(ValueError,match='injected'):
        joint.calibrate_joint(model,torch.randint(1,128,(2,32)))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    assert all(not m._forward_pre_hooks for m in model.modules())


def test_joint_rejects_invalid_targets():
    for half,s in [(0,.5),(16,0),(16,1)]:
        with pytest.raises(ValueError):fit_joint(torch.zeros(2,1,1),half,s)
