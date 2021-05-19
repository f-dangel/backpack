"""Module containing Permute module."""
from typing import Any

from torch import Tensor
from torch.nn import Module


class Permute(Module):
    """Module to permute a tensor."""

    def __init__(self, dims: Any):
        """Initialization.

        Args:
            dims: The desired ordering of dimensions.
        """
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, input: Tensor) -> Tensor:
        """Permutes the input tensor.

        Args:
            input: input tensor

        Returns:
            view with new ordering
        """
        return input.permute(self.dims)
