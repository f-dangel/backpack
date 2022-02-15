"""Contains ``DiagH`` extension for BackPACK's custom ``Pad`` module."""

from backpack.core.derivatives.pad import PadDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHPad(DiagHBaseModule):
    """``DiagH`` extension for ``backpack.custom_modules.pad.Pad``."""

    def __init__(self):
        """Pass derivatives for ``backpack.custom_modules.pad.Pad`` module."""
        super().__init__(PadDerivatives())
