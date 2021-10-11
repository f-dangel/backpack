"""Contains extensions for padding layers used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNZeroPad2d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.ZeroPad2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ZeroPad2d`` module."""
        super().__init__(ZeroPad2dDerivatives())
