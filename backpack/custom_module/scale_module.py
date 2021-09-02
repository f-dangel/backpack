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
            ValueError: if weight is no float
        """
        super().__init__()
        if not isinstance(weight, float):
            raise ValueError("Weight must be float.")
        self.weight: float = weight

    def forward(self, input: Tensor) -> Tensor:
        """Defines forward pass.

        Args:
            input: input

        Returns:
            product of input and weight
        """
        return input * self.weight
