"""Module containing Permute module."""
from typing import Any

from torch import Tensor
from torch.nn import Module


class Permute(Module):
    """Module to permute a tensor."""

    def __init__(self, *dims: Any, init_transpose: bool = False):
        """Initialization.

        This module supports two variants: permutation and transposition.
        If transposition should be used, a tuple (axis1, axis2) should be provided and
        init_transpose must be True.
        Internally, this is converted to a permutation in the first forward pass.

        Args:
            dims: The desired ordering of dimensions.
            init_transpose: If transpose parameters are provided. Default: False.
        """
        super().__init__()
        self.dims = dims
        self.init_transpose = init_transpose
        self._enforce_batch_axis_first()

    def forward(self, input: Tensor) -> Tensor:
        """Permutes the input tensor.

        Args:
            input: input tensor

        Returns:
            view with new ordering
        """
        self._convert_transpose_to_permute(input)
        return input.permute(self.dims)

    def _convert_transpose_to_permute(self, input: Tensor):
        """Converts the parameters of transpose to a permutation.

        Args:
            input: input tensor. Used to determine number of dimensions.
        """
        if self.init_transpose:
            permutation = list(range(input.dim()))
            permutation[self.dims[0]] = self.dims[1]
            permutation[self.dims[1]] = self.dims[0]
            self.dims = tuple(permutation)
            self.init_transpose = False

    def _enforce_batch_axis_first(self) -> None:
        batch_first = False
        if self.init_transpose:
            if 0 not in self.dims:
                batch_first = True
        else:
            if self.dims[0] == 0:
                batch_first = True
        if not batch_first:
            raise ValueError("Permute: Batch axis must be left unchanged!")
