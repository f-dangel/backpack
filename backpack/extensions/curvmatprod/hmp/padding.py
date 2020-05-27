from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPZeroPad2d(HMPBase):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
