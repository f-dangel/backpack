from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from backpack.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPZeroPad2d(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
