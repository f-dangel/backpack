from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .cmpbase import CMPBase


class CMPZeroPad2d(CMPBase):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
