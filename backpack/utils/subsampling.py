"""Utility functions to enable mini-batch subsampling in extensions."""
from typing import List

from torch import Tensor
from torch.nn import LSTM, RNN, Module, Sequential

from backpack.custom_module.permute import Permute


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


def get_batch_axis(module: Module, io_str: str) -> int:
    """Return the batch axis assumed by the module.

    For unknown modules the default axis is determined as ``0``.

    Args:
        module: A module.
        io_str: Name of the tensor stored as BackPACK IO. Must be ``'input0'`` or
            ``'output'``.

    Note:
        This method only inspects single modules and therefore cannot detect whether
        the batch axis has been modified by preceding ones. For instance for a ReLU
        module, the batch axis will always be detected as ``0``, although the layer
        still works if preceded by a ``Permute(0, 1)`` module, but would have batch
        axis ``1``.

    Returns:
        Batch axis

    Raises:
        ValueError: For invalid IO names.
    """
    valid_io_strs = ["input0", "output"]
    if io_str not in valid_io_strs:
        raise ValueError(f"io_str must be in {valid_io_strs}, got {io_str}.")

    batch_axis = 0

    if isinstance(module, (RNN, LSTM)):
        batch_axis = 0 if module.batch_first else 1
    elif isinstance(module, Permute):
        batch_axis = module.get_batch_axis(io_str)
    elif isinstance(module, Sequential):
        child_idx = {"input0": 0, "output": -1}[io_str]
        batch_axis = get_batch_axis(list(module.children())[child_idx], io_str)

    return batch_axis
