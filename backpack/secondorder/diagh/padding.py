from ...core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .diaghbase import DiagHBase


class DiagHZeroPad2d(DiagHBase, ZeroPad2dDerivatives):
    pass


EXTENSIONS = [
    DiagHZeroPad2d(),
]
