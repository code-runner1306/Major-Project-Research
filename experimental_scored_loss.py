"""Loss-only diagnostic path; skip vocabulary logits at ignored positions."""
import torch
from torch.nn import functional as F

from model import GatedDeltaNetLM


def scored_token_loss(model: GatedDeltaNetLM, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Same masked mean CE as GDN forward, including causal input history.

    This skips output projections, not context tokens or recurrent updates.
    Restricted to the exact base LM: subclasses may add positions or change
    forward semantics and must implement their own verified loss-only adapter.
    """
    if type(model) is not GatedDeltaNetLM:
        raise TypeError('Scored-token adapter supports the base GatedDeltaNetLM only')
    if tokens.ndim != 2 or targets.shape != tokens.shape:
        raise ValueError('tokens and targets must have matching [batch,time] shapes')
    hidden = model.tok_embeddings(tokens)
    for block in model.blocks:
        hidden = block(hidden)
    mask = targets != -100
    # No token is removed from the recurrent context. Only the expensive final
    # vocabulary projection is restricted to positions the objective scores.
    logits = model.output(model.norm(hidden)[mask])
    return F.cross_entropy(logits, targets[mask])
