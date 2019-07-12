from ...core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .kfacbase import KFACBase


class KFACZeroPad2d(KFACBase, ZeroPad2dDerivatives):
    pass


EXTENSIONS = [
    KFACZeroPad2d(),
]
