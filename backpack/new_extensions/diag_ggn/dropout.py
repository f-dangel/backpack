from backpack.core.derivatives.dropout import DropoutDerivatives
from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNDropout(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
