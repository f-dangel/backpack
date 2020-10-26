from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPZeroPad2d(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
