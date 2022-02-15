"""Contains ``DiagGGN{Exact, MC}`` extension for BackPACK's custom ``Pad`` module."""

from backpack.core.derivatives.pad import PadDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNPad(DiagGGNBaseModule):
    """``DiagGGN{Exact, MC}`` extension for ``backpack.custom_modules.pad.Pad``."""

    def __init__(self):
        """Pass derivatives for ``backpack.custom_modules.pad.Pad`` module."""
        super().__init__(PadDerivatives())
