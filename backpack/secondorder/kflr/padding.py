from ...core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .kflrbase import KFLRBase


class KFLRZeroPad2d(KFLRBase, ZeroPad2dDerivatives):
    pass


EXTENSIONS = [
    KFLRZeroPad2d(),
]
