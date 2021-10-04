"""Module containing Permute module."""
from typing import Any

from torch import Tensor
from torch.nn import Module


class Permute(Module):
    """Module to permute a tensor."""

    def __init__(self, *dims: Any, init_transpose: bool = False, batch_axis: int = 0):
        """Initialization.

        Args:
            dims: The desired ordering of dimensions.
            init_transpose: If transpose parameters are provided. Default: False.
            batch_axis: Which axis assumed to be the batch axis in a forward pass.
                Defaults to ``0``.
        """
        super().__init__()
        self.dims = dims
        self.init_transpose = init_transpose
        self.batch_axis = batch_axis

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

    def get_batch_axis(self, io_str: str) -> int:
        """Return the batch axis assumed by the module.

        Args:
            io_str: Name of the tensor. Must be ``'input0'`` or ``'output'``.

        Returns:
            Batch axis

        Raises:
            ValueError: For invalid IO names.
        """
        if io_str == "input0":
            return self.batch_axis
        elif io_str == "output":
            return self.dims.index(self.batch_axis)
        else:
            valid_io_strs = ["input0", "output"]
            raise ValueError(f"io_str must be in {valid_io_strs}, got {io_str}.")
