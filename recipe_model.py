"""Pinned Zoology backbone with a source-based, portable GDN mixer extension.

GDN equations/initialization follow FLA2705ab8 (MIT); see third_party/fla_reference.
The scalar->channel decay extension and pure-PyTorch executor are ours. This is
separate from the earlier simplified sigmoid-gate model and its checkpoints.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
import zlib

import torch
from torch import nn
from torch.nn import functional as F

from experimental_chunked_recurrence import chunked_recurrence
from recurrence import RecurrenceOptions
from third_party.zoology_iclr24.attention import MHA
from third_party.zoology_iclr24.model import LanguageModel

FLA_REVISION='2705ab8ee9a9086bcfa6d079858e4281e01215a1'


def published_l2norm(x):
    """FLA uses sum(x**2)+1e-6, not clamp(norm,min=eps)."""
    working=x.float()
    return (working*torch.rsqrt(working.square().sum(-1,keepdim=True)+1e-6)).to(x.dtype)


def serial_rectangular(q,k,v,alpha,beta,initial_state=None):
    """Independent serial decay-before-residual oracle, value-by-key state."""
    state=(q.new_zeros(q.size(0),q.size(2),v.size(-1),k.size(-1))
           if initial_state is None else initial_state)
    output=[]
    for index in range(q.size(1)):
        state=state*alpha[:,index].unsqueeze(-2)
        key=k[:,index].unsqueeze(-1)
        residual=v[:,index].unsqueeze(-1)-state@key
        state=state+beta[:,index].unsqueeze(-1)*residual@key.transpose(-1,-2)
        output.append((state@q[:,index].unsqueeze(-1)).squeeze(-1))
    return torch.stack(output,dim=1),state


class RecipeGDN(nn.Module):
    def __init__(self,d_model,layer_idx,variant='scalar',backend='reference',heads=1,chunk_size=16):
        super().__init__()
        if variant not in ('scalar','channel-wise') or backend not in ('reference','chunked') or d_model%heads:
            raise ValueError('Invalid recipe GDN settings')
        self.width,self.heads,self.layer_idx=d_model,heads,layer_idx
        self.head_k=d_model//heads
        self.head_v=2*self.head_k
        self.gate_width=1 if variant=='scalar' else self.head_k
        self.backend,self.chunk_size=backend,chunk_size
        self.q_proj=nn.Linear(d_model,d_model,bias=False)
        self.k_proj=nn.Linear(d_model,d_model,bias=False)
        self.v_proj=nn.Linear(d_model,2*d_model,bias=False)
        self.a_proj=nn.Linear(d_model,heads*self.gate_width,bias=False)
        self.b_proj=nn.Linear(d_model,heads,bias=False)
        self.q_conv=nn.Conv1d(d_model,d_model,4,groups=d_model,bias=False)
        self.k_conv=nn.Conv1d(d_model,d_model,4,groups=d_model,bias=False)
        self.v_conv=nn.Conv1d(2*d_model,2*d_model,4,groups=2*d_model,bias=False)
        # Channel extension starts with the same per-head base timescale, while
        # projection rows can adapt independently. These parameters never decay.
        rate=torch.empty(heads,1).uniform_(0,16).clamp_min(1e-8)
        dt=torch.exp(torch.rand(heads,1)*(math.log(.1)-math.log(.001))+math.log(.001))
        inverse=dt+torch.log(-torch.expm1(-dt))
        self.A_log=nn.Parameter(rate.log().expand(heads,self.gate_width).clone())
        self.dt_bias=nn.Parameter(inverse.expand(heads,self.gate_width).clone())
        self.A_log._no_weight_decay=True
        self.dt_bias._no_weight_decay=True
        self.g_proj=nn.Linear(d_model,2*d_model,bias=False)
        self.o_norm=nn.Parameter(torch.ones(self.head_v))
        self.o_proj=nn.Linear(2*d_model,d_model,bias=False)

    def decay(self,x):
        raw=self.a_proj(x).view(x.size(0),x.size(1),self.heads,self.gate_width).float()
        return torch.exp(-self.A_log.float().exp()*F.softplus(raw+self.dt_bias.float()))

    def forward(self,x):
        def projected(projection,convolution,width):
            raw=projection(x).transpose(1,2)
            return F.silu(convolution(F.pad(raw,(3,0))).transpose(1,2)).reshape(x.size(0),x.size(1),self.heads,width)
        q=published_l2norm(projected(self.q_proj,self.q_conv,self.head_k))
        k=published_l2norm(projected(self.k_proj,self.k_conv,self.head_k))
        v=projected(self.v_proj,self.v_conv,self.head_v)
        alpha=self.decay(x)
        beta=self.b_proj(x).view(x.size(0),x.size(1),self.heads,1).sigmoid()
        # FLA's default output query scale is key_width**-0.5.
        q=q*self.head_k**-.5
        if self.backend=='chunked':
            output,_=chunked_recurrence(q,k,v,alpha,beta,options=RecurrenceOptions(
                chunk_size=self.chunk_size,state_accumulator_dtype='float32'))
        else:
            output,_=serial_rectangular(q,k,v,alpha,beta)
        normalized=output.float()*torch.rsqrt(output.float().square().mean(-1,keepdim=True)+1e-5)
        gated=normalized*self.o_norm*F.silu(self.g_proj(x).view_as(output))
        return self.o_proj(gated.flatten(-2))


class Factory:
    def __init__(self,constructor,**kwargs):
        self.constructor,self.kwargs=constructor,kwargs
    def instantiate(self,**kwargs):
        return self.constructor(**kwargs,**self.kwargs)


class RecipeLM(nn.Module):
    def __init__(self,architecture='scalar',vocabulary=8192,length=64,width=128,backend='reference',seed=42):
        super().__init__()
        if architecture not in ('scalar','channel-wise','attention'):
            raise ValueError('Unknown architecture')
        attention=architecture=='attention'
        torch.manual_seed(seed)
        self.config=SimpleNamespace(d_model=width,vocab_size=vocabulary,
            n_layers=2 if attention else 4,max_position_embeddings=length if attention else 0,
            learnable_word_embeddings=True,block_type='TransformerBlock',embed_dropout=.1,
            resid_dropout=0.,drop_path=0.,layer_norm_epsilon=1e-5,pad_vocab_size_multiple=1,
            state_mixer=Factory(nn.Identity),sequence_mixer=(Factory(MHA,num_heads=1,dropout=.1)
            if attention else Factory(RecipeGDN,variant=architecture,backend=backend)))
        self.network=LanguageModel(self.config)
        if not attention:
            self._paired_initialization(seed)

    def _paired_initialization(self,seed):
        """Stable module streams prevent gate shape from changing common weights."""
        def generator(name):
            return torch.Generator().manual_seed(zlib.crc32(f'{seed}:{name}'.encode())&0x7fffffff)
        with torch.no_grad():
            for name,module in self.network.named_modules():
                if isinstance(module,(nn.Linear,nn.Embedding)):
                    if name.endswith('a_proj'):
                        rows=[torch.randn(module.in_features,generator=generator(f'{name}:{row}'))*.02
                              for row in range(module.out_features)]
                        module.weight.copy_(torch.stack(rows))
                    else:
                        std=.02/math.sqrt(2*self.config.n_layers) if name.endswith('out_proj') else .02
                        nn.init.normal_(module.weight,std=std,generator=generator(name))
                    if getattr(module,'bias',None) is not None:
                        module.bias.zero_()
                elif isinstance(module,nn.Conv1d):
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(generator(name).initial_seed())
                        module.reset_parameters()
                elif isinstance(module,RecipeGDN):
                    gen=generator(name+':timescale')
                    rate=torch.rand(module.heads,1,generator=gen).mul(16).clamp_min(1e-8)
                    dt=torch.exp(torch.rand(module.heads,1,generator=gen)*(math.log(.1)-math.log(.001))+math.log(.001))
                    module.A_log.copy_(rate.log().expand_as(module.A_log))
                    module.dt_bias.copy_((dt+torch.log(-torch.expm1(-dt))).expand_as(module.dt_bias))

    def forward(self,tokens,targets=None):
        logits=self.network(tokens)
        loss=None if targets is None else F.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=-100)
        return logits,loss

    def loss(self,tokens,targets,scored_only=True):
        hidden=self.network.backbone(tokens)
        if scored_only:
            mask=targets!=-100
            return F.cross_entropy(self.network.lm_head(hidden[mask]),targets[mask])
        return F.cross_entropy(self.network.lm_head(hidden).flatten(0,1),targets.flatten(),ignore_index=-100)

    def scored_logits(self,tokens,mask):
        return self.network.lm_head(self.network.backbone(tokens)[mask])

    def optimizer(self,learning_rate):
        # Parameter attributes can disappear on deepcopy/device conversion.
        # Structural names keep timescale parameters exempt in every worker.
        exempt=[p for name,p in self.named_parameters() if name.endswith(('A_log','dt_bias'))]
        regular=[p for name,p in self.named_parameters() if not name.endswith(('A_log','dt_bias'))]
        return torch.optim.AdamW([{'params':regular,'weight_decay':.1},
                                 {'params':exempt,'weight_decay':0.}],lr=learning_rate)
