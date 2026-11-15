import copy

import pytest
import torch

from config import ModelConfig
from experimental_chunked_recurrence import install_candidate
from experimental_scored_loss import scored_token_loss
from model import GatedDeltaNetLM
from recurrence import RecurrenceOptions


@pytest.mark.parametrize('variant', ['scalar', 'channel-wise'])
@pytest.mark.parametrize('backend', ['reference', 'chunked'])
def test_scored_loss_and_every_parameter_gradient_match_full_logits(variant, backend):
    torch.manual_seed(871)
    model = GatedDeltaNetLM(ModelConfig('loss-check',16,2,2,32,'pure_gdn'),variant,128,
                           RecurrenceOptions(state_accumulator_dtype='float32'),study_init_seed=871)
    if backend == 'chunked':
        install_candidate(model,chunk_size=4)
    other = copy.deepcopy(model)
    tokens = torch.randint(0,128,(2,19))
    labels = torch.full_like(tokens,-100)
    labels[0,[4,7,12]]=torch.tensor([70,75,85])
    labels[1,[3,16]]=torch.tensor([81,91])
    _,expected = model(tokens,labels)
    actual = scored_token_loss(other,tokens,labels)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    expected.backward()
    actual.backward()
    for (name,p),(other_name,other_p) in zip(model.named_parameters(),other.named_parameters()):
        assert name == other_name
        assert p.grad is not None and other_p.grad is not None
        torch.testing.assert_close(other_p.grad,p.grad,atol=2e-6,rtol=2e-5)


def test_scored_loss_rejects_forward_overriding_subclass():
    class DifferentLM(GatedDeltaNetLM):
        pass
    model=DifferentLM(ModelConfig('unsafe-loss',8,1,1,16,'pure_gdn'),'scalar',32)
    with pytest.raises(TypeError):
        scored_token_loss(model,torch.zeros(1,8,dtype=torch.long),torch.zeros(1,8,dtype=torch.long))
