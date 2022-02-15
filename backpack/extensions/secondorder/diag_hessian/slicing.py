"""Contains ``DiagH`` extension for BackPACK's custom ``Slicing`` module."""

from backpack.core.derivatives.slicing import SlicingDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHSlicing(DiagHBaseModule):
    """``DiagH`` extension for ``backpack.custom_modules.slicing.Slicing``."""

    def __init__(self):
        """Pass derivatives for ``backpack.custom_modules.slicing.Slicing`` module."""
        super().__init__(SlicingDerivatives())
