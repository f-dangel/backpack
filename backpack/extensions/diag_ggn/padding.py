from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNZeroPad2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
