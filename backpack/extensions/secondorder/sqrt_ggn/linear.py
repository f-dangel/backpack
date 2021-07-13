"""Contains extension for the linear layer used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNLinear(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Linear`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Linear`` module."""
        super().__init__(LinearDerivatives(), params=["bias", "weight"])
