"""Contains extensions for dropout layers used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNDropout(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Dropout`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Dropout`` module."""
        super().__init__(DropoutDerivatives())
