from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPZeroPad2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
