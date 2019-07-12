from ...core.derivatives.zeropad2d import ZeroPad2dDerivatives
from .diagggnbase import DiagGGNBase


class DiagGGNZeroPad2d(DiagGGNBase, ZeroPad2dDerivatives):
    pass


EXTENSIONS = [
    DiagGGNZeroPad2d(),
]
