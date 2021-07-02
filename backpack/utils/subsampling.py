"""Utility functions to enable mini-batch subsampling in extensions."""
from typing import List

from torch import Tensor
from torch.nn import Module


def subsample(module: Module, tensor_str: str, subsampling: List[int] = None) -> Tensor:
    """Return a sub-sampled tensor stored in a module.

    Args:
        module: Layer with BackPACK IO stored in a forward pass.
        tensor_str: Argument under which the tensor is stored in the module.
        subsampling: Indices of samples that are sliced along the batch dimension.
            Defaults to ``None`` (use all samples).

    Returns:
        Tensor that is sub-sampled along the batch dimension such that only the
        active samples are contained.

    Raises:
        NotImplementedError: If batch axis of the module is not first.
    """
    tensor = getattr(module, tensor_str)

    if subsampling is None:
        return tensor
    else:
        batch_axis = _get_batch_axis(module)
        if batch_axis != 0:
            raise NotImplementedError(f"Only batch axis 0 supported. Got {batch_axis}.")
        else:
            return tensor[subsampling]


def _get_batch_axis(module: Module) -> int:
    """Return the batch axis assumed by a layer.

    Args:
        module: Layer.

    Returns:
        Batch axis.
    """
    return 0
