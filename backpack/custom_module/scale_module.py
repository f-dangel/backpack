"""Contains ScaleModule."""
from torch import Tensor
from torch.nn import Module


class ScaleModule(Module):
    """Scale Module scales the input by a constant."""

    def __init__(self, weight: float = 1.0):
        """Store scalar weight.

        Args:
            weight: Initial value for weight. Defaults to 1.0.

        Raises:
            AssertionError: if weight is no float
        """
        super().__init__()
        assert isinstance(weight, float)
        self.weight: float = weight

    def forward(self, input: Tensor) -> Tensor:
        """Defines forward pass.

        Args:
            input: input

        Returns:
            product of input and weight
        """
        assert isinstance(input, Tensor)
        return input * self.weight
