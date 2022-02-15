"""Holds ``SqrtGGN{Exact, MC}`` extension for BackPACK's custom ``Slicing`` module."""

from backpack.core.derivatives.slicing import SlicingDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNSlicing(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` for ``backpack.custom_modules.slicing.Slicing``."""

    def __init__(self):
        """Pass derivatives for ``backpack.custom_modules.pad.Pad`` module."""
        super().__init__(SlicingDerivatives())
