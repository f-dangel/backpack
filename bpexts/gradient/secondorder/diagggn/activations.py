from ...derivatives.relu import ReLUDerivatives
from ...derivatives.tanh import TanhDerivatives
from ...derivatives.sigmoid import SigmoidDerivatives
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
