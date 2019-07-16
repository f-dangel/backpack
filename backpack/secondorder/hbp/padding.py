from ...core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .hbpbase import HBPBase


class HBPZeroPad2d(HBPBase, ZeroPad2dDerivatives):
    pass


EXTENSIONS = [
    HBPZeroPad2d(),
]
