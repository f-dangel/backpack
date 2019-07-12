from ..core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .cmpbase import CMPBase


class CMPZeroPad2d(CMPBase, ZeroPad2dDerivatives):
    pass


EXTENSIONS = [
    CMPZeroPad2d(),
]
