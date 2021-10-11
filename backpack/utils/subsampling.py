"""Utility functions to enable mini-batch subsampling in extensions."""
from typing import List

from torch import Tensor


def subsample(tensor: Tensor, dim: int = 0, subsampling: List[int] = None) -> Tensor:
    """Select samples from a tensor along a dimension.

    Args:
        tensor: Tensor to select from.
        dim: Selection dimension. Defaults to ``0``.
        subsampling: Indices of samples that are sliced along the dimension.
            Defaults to ``None`` (use all samples).

    Returns:
        Tensor of same rank that is sub-sampled along the dimension.
    """
    if subsampling is None:
        return tensor
    else:
        return tensor[(slice(None),) * dim + (subsampling,)]
