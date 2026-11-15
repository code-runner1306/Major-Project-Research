import copy

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experimental_chunked_recurrence import chunked_recurrence
from recipe_data import audit_mqar, build_data, mqar, split_overlap
from recipe_model import RecipeLM, published_l2norm, serial_rectangular
from recurrence import RecurrenceOptions
from third_party.zoology_iclr24.mqar import _mqar


@pytest.mark.parametrize('seed',[7,42])
@pytest.mark.parametrize('vocabulary',[256,8192])
def test_memory_bounded_generator_is_byte_identical_to_pinned_source(seed,vocabulary):
    state=np.random.get_state()
    try:
        expected=_mqar(vocab_size=vocabulary,num_examples=64,input_seq_len=64,seed=seed,
                       num_kv_pairs=4,random_non_queries=False)
    finally:
        np.random.set_state(state)
    before=np.random.get_state()
    result=mqar(examples=64,vocabulary=vocabulary,length=64,pairs=4,seed=seed)
    for actual,reference in zip(result,expected):
        assert torch.equal(actual,reference)
    assert np.array_equal(before[1],np.random.get_state()[1])
    assert audit_mqar(*result)['passed']
    corrupted=result[1].clone()
    position=(corrupted!=-100).nonzero()[0]
    corrupted[tuple(position)]=0
    assert not audit_mqar(result[0],corrupted)['passed']


def test_published_l2norm_zero_vector_has_the_upstream_finite_derivative():
    values=torch.zeros(2,4,requires_grad=True)
    result=published_l2norm(values)
    result.sum().backward()
    assert torch.equal(result,torch.zeros_like(result))
    torch.testing.assert_close(values.grad,torch.full_like(values,1000.))


def test_split_overlap_checks_content_not_split_names():
    train=build_data('mqar',16,42,split='train',vocabulary=128,length=32)
    identical=build_data('mqar',16,42,split='validation',vocabulary=128,length=32)
    independent=build_data('mqar',16,43,split='validation',vocabulary=128,length=32)
    assert split_overlap(train,identical)==16
    assert split_overlap(train,independent)==0


@pytest.mark.parametrize('gate_width',[1,5])
def test_rectangular_state_chunked_output_and_all_input_gradients(gate_width):
    torch.manual_seed(17)
    q,k=[F.normalize(torch.randn(2,11,2,5,dtype=torch.float64),dim=-1) for _ in range(2)]
    v=torch.randn(2,11,2,7,dtype=torch.float64)
    alpha=torch.rand(2,11,2,gate_width,dtype=torch.float64)
    alpha[:,4]=0
    beta=torch.rand(2,11,2,1,dtype=torch.float64)
    initial=torch.randn(2,2,7,5,dtype=torch.float64)
    inputs=[x.requires_grad_() for x in (q,k,v,alpha,beta,initial)]
    expected,state=serial_rectangular(*inputs)
    actual,actual_state=chunked_recurrence(*inputs,options=RecurrenceOptions(chunk_size=4))
    torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(actual_state,state,atol=1e-11,rtol=1e-11)
    ref_grad=torch.autograd.grad(expected.square().sum()+state.square().sum(),inputs)
    grad=torch.autograd.grad(actual.square().sum()+actual_state.square().sum(),inputs)
    for actual,expected in zip(grad,ref_grad):
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('architecture',['scalar','channel-wise','attention'])
def test_pinned_backbone_causal_and_scored_loss_gradient_equivalence(architecture):
    torch.set_num_threads(1)
    reference=RecipeLM(architecture,vocabulary=128,length=32,width=16,seed=9)
    reference.eval()  # Disable the published dropout for paired comparisons.
    candidate=copy.deepcopy(reference)
    tokens,labels=mqar(examples=2,vocabulary=128,length=32,pairs=4,seed=9)
    logits,expected=reference(tokens,labels)
    actual=candidate.loss(tokens,labels)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
    expected.backward()
    actual.backward()
    for (_,p),(_,other) in zip(reference.named_parameters(),candidate.named_parameters()):
        assert p.grad is not None and other.grad is not None
        torch.testing.assert_close(other.grad,p.grad,atol=2e-5,rtol=2e-4)
    changed=tokens.clone()
    changed[:,20:]=torch.randint(1,128,changed[:,20:].shape)
    with torch.no_grad():
        changed_logits,_=reference(changed)
    torch.testing.assert_close(logits[:,:20],changed_logits[:,:20],atol=1e-6,rtol=1e-5)


def test_gate_shape_does_not_change_common_weights_or_base_timescales():
    scalar=RecipeLM('scalar',vocabulary=128,length=32,width=16,seed=19)
    channel=RecipeLM('channel-wise',vocabulary=128,length=32,width=16,seed=19)
    for name,value in scalar.state_dict().items():
        other=channel.state_dict()[name]
        if name.endswith(('a_proj.weight','A_log','dt_bias')):
            continue
        assert torch.equal(value,other),name
    for left,right in zip(scalar.network.backbone.layers,channel.network.backbone.layers):
        assert torch.equal(left.sequence_mixer.A_log.expand_as(right.sequence_mixer.A_log),right.sequence_mixer.A_log)
        assert torch.equal(left.sequence_mixer.dt_bias.expand_as(right.sequence_mixer.dt_bias),right.sequence_mixer.dt_bias)
    optimizer=channel.optimizer(1e-3)
    exempt={id(p) for group in optimizer.param_groups if group['weight_decay']==0 for p in group['params']}
    assert all(id(p) in exempt for name,p in channel.named_parameters() if name.endswith(('A_log','dt_bias')))


@pytest.mark.parametrize('architecture',['scalar','channel-wise'])
def test_recipe_chunked_and_serial_five_optimizer_updates(architecture):
    reference=RecipeLM(architecture,vocabulary=128,length=32,width=16,seed=33)
    candidate=copy.deepcopy(reference)
    for layer in candidate.network.backbone.layers:
        layer.sequence_mixer.backend='chunked'
        layer.sequence_mixer.chunk_size=4
    reference.eval()
    candidate.eval()
    optimizers=[m.optimizer(1e-3) for m in (reference,candidate)]
    tokens,labels=mqar(examples=2,vocabulary=128,length=32,pairs=4,seed=33)
    for _ in range(5):
        for model,optimizer in zip((reference,candidate),optimizers):
            optimizer.zero_grad(set_to_none=True)
            loss=model.loss(tokens,labels)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        for actual,expected in zip(candidate.parameters(),reference.parameters()):
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)


@pytest.mark.parametrize('history',['answer_free','teacher_forced'])
def test_batched_overwrite_metrics_match_existing_evaluator_and_erasure(history):
    from config import RetrievalConfig
    from recipe_evaluation import evaluate
    from retrieval import RetrievalDataset,evaluate_retrieval
    data=build_data('overwrite',6,177,split='validation',vocabulary=128,length=32)
    config=RetrievalConfig(vocabulary_size=128,sequence_lengths=(33,),key_counts=(4,),
        query_distances=(8,),overwrite_counts=(1,),generator_version='mqar-overwrite-v4')
    dataset=RetrievalDataset(task='overwrite',examples=6,config=config,seed=177,split='validation')

    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs=[]
        def forward(self,tokens):
            logits=torch.zeros(*tokens.shape,128)
            for row,sequence in enumerate(tokens.tolist()):
                writes={}
                for position,key in enumerate(sequence):
                    if 2<=key<64:
                        if position and sequence[position-1]==0:
                            logits[row,position,writes[key]]=1
                        if position+1<len(sequence) and sequence[position+1]>=64:
                            writes[key]=sequence[position+1]
            return logits,None
        def scored_logits(self,tokens,mask):
            self.inputs.append(tokens.clone())
            return self(tokens)[0][mask]

    model=Oracle()
    expected=evaluate_retrieval(model,dataset,torch.device('cpu'),history_conditioning=history)
    actual=evaluate(model,data,torch.device('cpu'),batch=4,history=history)
    for name in ('exact_query_accuracy','correct','total','first_query_accuracy','later_query_accuracy',
                 'latest_value_accuracy','unrelated_association_retention','stale_value_error_rate',
                 'example_accuracy','evaluated_examples'):
        assert actual[name]==expected[name],name
    assert actual['exact_query_accuracy']==1.
    fed=torch.cat(model.inputs)
    if history=='answer_free':
        assert (fed[data.erase_answers]==1).all()
    else:
        assert torch.equal(fed,data.tokens)


@pytest.mark.parametrize('family',['mqar','overwrite'])
def test_batched_real_model_scores_do_not_depend_on_batch_partition(family):
    from recipe_evaluation import evaluate
    data=build_data(family,6,123,split='validation',vocabulary=128,length=32)
    model=RecipeLM('scalar',vocabulary=128,length=32,width=16,seed=123)
    first=evaluate(model,data,torch.device('cpu'),batch=1)
    batched=evaluate(model,data,torch.device('cpu'),batch=4)
    for key in ('correct','total','first_query_accuracy','latest_value_accuracy','unrelated_association_retention'):
        assert first[key]==batched[key]
    assert first['loss']==pytest.approx(batched['loss'],abs=1e-5)
