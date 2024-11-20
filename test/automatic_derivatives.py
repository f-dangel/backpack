"""Define derivatives of layers computed via autodiff."""

from typing import Callable, Optional

from torch import Tensor
from torch.nn import Linear, ReLU, Sigmoid
from torch.nn.functional import linear, relu, sigmoid

from backpack.core.derivatives.automatic import AutomaticDerivatives


class LinearAutomaticDerivatives(AutomaticDerivatives):
    """Automatic derivatives for ``torch.nn.Linear``."""

    @staticmethod
    def as_functional(
        module: Linear,
    ) -> Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]:
        """Return the linear layer's forward pass function.

        Args:
            module: A linear layer.

        Returns:
            The linear layer's forward pass function.
        """
        return linear


class ReLUAutomaticDerivatives(AutomaticDerivatives):
    """Automatic derivatives for ``torch.nn.ReLU``."""

    @staticmethod
    def as_functional(module: ReLU) -> Callable[[Tensor], Tensor]:
        """Return the ReLU layer's forward pass function.

        Args:
            module: A ReLU layer.

        Returns:
            The ReLU layer's forward pass function.
        """
        return relu


class SigmoidAutomaticDerivatives(AutomaticDerivatives):
    """Automatic derivatives for ``torch.nn.Sigmoid``."""

    @staticmethod
    def as_functional(module: Sigmoid) -> Callable[[Tensor], Tensor]:
        """Return the Sigmoid layer's forward pass function.

        Args:
            module: A Sigmoid layer.

        Returns:
            The Sigmoid layer's forward pass function.
        """
        return sigmoid
