from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHZeroPad2d(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
