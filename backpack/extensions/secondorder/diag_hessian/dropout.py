from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHDropout(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
