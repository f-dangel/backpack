"""Contains extensions for the flatten layer used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNFlatten(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Flatten`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Flatten`` module."""
        super().__init__(FlattenDerivatives())
