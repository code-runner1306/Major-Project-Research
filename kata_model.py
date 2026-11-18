"""Model used to replicate the Gated DeltaNet row of KATA (arXiv 2607.17419), Table 2, dh=64.

Appendix F.3/F.4: the canonical two-layer Zoology TransformerBlock stack with state mixer Identity and sequence
mixer Hybrid(BaseConv, mixer). Zoology's Hybrid picks one module per layer, so layer 0 is BaseConv(k=3) and layer 1
is Gated DeltaNet. One short convolution therefore precedes the sequence operator, no positional embeddings, tied LM
head, vocabulary 8192, Zoology N(0, 0.02^2) initialisation with GPT-2 residual scaling on the output projection.

Gated DeltaNet is FLA's mixer "out of the box" with use_gate=False, short convolution k=4, q/k L2 norm and value
expansion 2 (F.4). The Table 2 parameter counts identify the dh=64 row as d_model=128 with 2 heads:
our count is 1,167,876 against the reported 1,167,878. F.4's prose says NH=1, but the counts in Table 2 for every
GDN and softmax row match two heads of size dh with d_model = 2 * dh, so we follow the table.
The GDN equations are our portable implementation of FLA's (recipe_model.RecipeGDN); recipe_model.py is not modified.
"""
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from recipe_model import RecipeGDN, published_l2norm, serial_rectangular
from experimental_chunked_recurrence import chunked_recurrence
from recurrence import RecurrenceOptions
from third_party.zoology_iclr24.model import LanguageModel

PAPER_PARAMETERS = 1167878


class ShortConvolution(nn.Module):
    """Zoology's dimension-last depthwise causal convolution (zoology/mixers/convolution.py, revision de4e258)."""

    def __init__(self, d_model, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=kernel_size - 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., :x.size(1)].transpose(1, 2)


class BaseConv(nn.Module):
    """Zoology BaseConv (zoology/mixers/base_conv.py, revision de4e258) with a short kernel and use_act=False."""

    def __init__(self, d_model, kernel_size=3, **kwargs):
        super().__init__()
        self.projection = nn.Linear(d_model, d_model)
        self.conv = ShortConvolution(d_model, kernel_size)

    def forward(self, u):
        return self.conv(u) * self.projection(u) + u


class GDNNoGate(RecipeGDN):
    """RecipeGDN with FLA's use_gate=False: per-head RMSNorm of the output, then the output projection."""

    def __init__(self, d_model, layer_idx, heads=2, backend='chunked'):
        super().__init__(d_model, layer_idx, variant='scalar', backend=backend, heads=heads)
        del self.g_proj

    def forward(self, x):
        def projected(projection, convolution, width):
            raw = projection(x).transpose(1, 2)
            return F.silu(convolution(F.pad(raw, (3, 0))).transpose(1, 2)).reshape(x.size(0), x.size(1), self.heads, width)
        q = published_l2norm(projected(self.q_proj, self.q_conv, self.head_k))
        k = published_l2norm(projected(self.k_proj, self.k_conv, self.head_k))
        v = projected(self.v_proj, self.v_conv, self.head_v)
        alpha = self.decay(x)
        beta = self.b_proj(x).view(x.size(0), x.size(1), self.heads, 1).sigmoid()
        q = q * self.head_k ** -.5
        if self.backend == 'chunked':
            output, _ = chunked_recurrence(q, k, v, alpha, beta, options=RecurrenceOptions(
                chunk_size=self.chunk_size, state_accumulator_dtype='float32'))
        else:
            output, _ = serial_rectangular(q, k, v, alpha, beta)
        normalized = output.float() * torch.rsqrt(output.float().square().mean(-1, keepdim=True) + 1e-5) * self.o_norm
        return self.o_proj(normalized.flatten(-2))


class HybridFactory:
    """Zoology Hybrid(BaseConv, mixer): one module per layer index."""

    def __init__(self, heads, backend):
        self.heads, self.backend = heads, backend

    def instantiate(self, d_model, layer_idx):
        if layer_idx == 0:
            return BaseConv(d_model, kernel_size=3)
        return GDNNoGate(d_model, layer_idx, heads=self.heads, backend=self.backend)


class IdentityFactory:
    def instantiate(self, **kwargs):
        return nn.Identity()


class KataLM(nn.Module):
    def __init__(self, d_model=128, heads=2, vocabulary=8192, backend='chunked', seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.config = SimpleNamespace(d_model=d_model, vocab_size=vocabulary, n_layers=2, max_position_embeddings=0,
                                      learnable_word_embeddings=True, block_type='TransformerBlock', embed_dropout=0.,
                                      resid_dropout=0., drop_path=0., layer_norm_epsilon=1e-5, pad_vocab_size_multiple=1,
                                      state_mixer=IdentityFactory(), sequence_mixer=HybridFactory(heads, backend))
        self.network = LanguageModel(self.config)
        with torch.no_grad():  # GPT-2 residual scaling on W_o (F.4); Zoology's helper only matches 'out_proj' names
            for module in self.modules():
                if isinstance(module, GDNNoGate):
                    nn.init.normal_(module.o_proj.weight, std=.02 / math.sqrt(2 * self.config.n_layers))

    def loss(self, tokens, targets):
        hidden = self.network.backbone(tokens)
        mask = targets != -100
        return F.cross_entropy(self.network.lm_head(hidden[mask]), targets[mask])

    def scored_logits(self, tokens, mask):
        return self.network.lm_head(self.network.backbone(tokens)[mask])

    def optimizer(self, learning_rate):
        exempt = [p for n, p in self.named_parameters() if n.endswith(('A_log', 'dt_bias'))]
        regular = [p for n, p in self.named_parameters() if not n.endswith(('A_log', 'dt_bias'))]
        return torch.optim.AdamW([{'params': regular, 'weight_decay': .1}, {'params': exempt, 'weight_decay': 0.}],
                                 lr=learning_rate, betas=(.9, .95))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
