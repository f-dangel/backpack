"""Module version of ``torch.nn.functional.pad``."""

from typing import Sequence

from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad


class Pad(Module):
    """Module version of ``torch.nn.functional.pad`` (N-dimensional padding)."""

    def __init__(self, pad: Sequence[int], mode: str = "constant", value: float = 0.0):
        """Store padding hyperparameters.

        See https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html.

        Args:
            pad: Tuple of even length specifying the padding.
            mode: Padding mode. Default ``'constant'``.
            value: Fill value for constant padding. Default ``0.0``.
        """
        super().__init__()
        self.pad = pad
        self.mode = mode
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        """Pad the input tensor.

        Args:
            input: Input tensor.

        Returns:
            Padded input tensor.
        """
        return pad(input, self.pad, mode=self.mode, value=self.value)
