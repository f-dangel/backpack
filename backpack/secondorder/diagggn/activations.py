from ...core.derivatives.relu import ReLUDerivatives
from ...core.derivatives.tanh import TanhDerivatives
from ...core.derivatives.sigmoid import SigmoidDerivatives
from .diagggnbase import DiagGGNBase


class DiagGGNReLU(DiagGGNBase, ReLUDerivatives):
    pass


class DiagGGNSigmoid(DiagGGNBase, TanhDerivatives):
    pass


class DiagGGNTanh(DiagGGNBase, SigmoidDerivatives):
    pass


EXTENSIONS = [
    DiagGGNReLU(),
    DiagGGNSigmoid(),
    DiagGGNTanh(),
]
