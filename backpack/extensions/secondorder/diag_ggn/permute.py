"""Module defining DiagGGNPermute."""
from backpack.core.derivatives.permute import PermuteDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNPermute(DiagGGNBaseModule):
    """DiagGGN extension of Permute."""

    def __init__(self):
        """Initialize."""
        super().__init__(derivatives=PermuteDerivatives())
