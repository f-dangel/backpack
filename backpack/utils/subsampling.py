"""Utility functions to enable mini-batch subsampling in extensions."""
from typing import List

from torch import Tensor
from torch.nn import LSTM, RNN, Module


def subsample(tensor: Tensor, dim: int = 0, subsampling: List[int] = None) -> Tensor:
    """Select samples from a tensor along a dimension.

    Args:
        tensor: Tensor to select from.
        dim: Selection dimension. Defaults to ``0``.
        subsampling: Indices of samples that are sliced along the dimension.
            Defaults to ``None`` (use all samples).

    Returns:
        Tensor of same rank that is sub-sampled along the dimension.

    Raises:
        NotImplementedError: If dimension differs from ``0`` or ``1``.
    """
    if subsampling is None:
        return tensor
    else:
        if dim == 0:
            return tensor[subsampling]
        elif dim == 1:
            return tensor[:, subsampling]
        else:
            raise NotImplementedError(f"Only supports dim = 0,1 . Got {dim}.")


def get_batch_axis(module: Module) -> int:
    """Return the batch axis assumed by the module.

    Args:
        module: A module.

    Returns:
        Batch axis
    """
    if isinstance(module, (RNN, LSTM)):
        if module.batch_first:
            return 0
        else:
            return 1
    else:
        return 0
