"""Contains ScaleModule."""
import torch


class ScaleModule(torch.nn.Module):
    """Scale Module scales the input by a constant.."""

    def __init__(self, weight=1.0):
        """Store scalar weight.

        Args:
            weight(float, optional): Initial value for weight. Defaults to 2.0.
        """
        super().__init__()
        self.weight = weight

    def forward(self, input):
        """Defines forward pass.

        Args:
            input(torch.Tensor): input

        Returns:
            torch.Tensor: product of input and weight
        """
        return input * self.weight
