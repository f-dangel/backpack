"""Custom module to perform tensor slicing."""

from typing import Tuple, Union

from torch import Tensor
from torch.nn import Module


class Slicing(Module):
    """Module that slices a tensor."""

    def __init__(self, slice_info: Tuple[Union[slice, int]]):
        """Store the slicing object.

        Args:
            slice_info: Argument that is passed to the slicing operator in the
                forward pass.
        """
        super().__init__()
        self.slice_info = slice_info

    def forward(self, input: Tensor) -> Tensor:
        """Slice the input tensor.

        Args:
            input: Input tensor.

        Returns:
            Sliced input tensor.
        """
        return input[self.slice_info]
