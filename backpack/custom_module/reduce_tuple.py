"""Module containing ReduceTuple module."""
from typing import Union

from torch import Tensor
from torch.nn import Module


class ReduceTuple(Module):
    """Module reducing tuple input."""

    def __init__(self, index: int = 0):
        """Initialization.

        Args:
            index: which element to choose
        """
        super().__init__()
        self.index = index

    def forward(self, input: tuple) -> Union[tuple, Tensor]:
        """Reduces the tuple.

        Args:
            input: the tuple of data

        Returns:
            the selected element
        """
        return input[self.index]
