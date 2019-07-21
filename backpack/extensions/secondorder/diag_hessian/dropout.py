from backpack.core.derivatives.dropout import DropoutDerivatives
from .diag_h_base import DiagHBaseModule


class DiagHDropout(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
