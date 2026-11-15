"""Dependency-only adapters for the pinned, zero-drop-path experiment."""
from torch import nn


class StochasticDepth(nn.Identity):
    def __init__(self, p, mode):
        if p != 0 or mode != 'row':
            raise ValueError('Vendored recipe supports only its original drop_path=0 setting')
        super().__init__()


def rearrange(tensor, pattern, **sizes):
    """The two reshape patterns used by the pinned attention implementation."""
    if pattern == '... (three h d) -> ... three h d':
        width = sizes['d']
        return tensor.reshape(*tensor.shape[:-1], sizes['three'], tensor.shape[-1]//(sizes['three']*width), width)
    if pattern == '... h d -> ... (h d)':
        return tensor.flatten(-2)
    raise ValueError(f'Unsupported pinned attention reshape: {pattern}')
