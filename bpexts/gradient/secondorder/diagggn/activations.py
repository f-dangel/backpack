from ...jacobians.relu import ReLUJacobian
from ...jacobians.tanh import TanhJacobian
from ...jacobians.sigmoid import SigmoidJacobian
from .diagggnbase import DiagGGNBase


class DiagGGNReLU(DiagGGNBase, ReLUJacobian):
    pass


class DiagGGNSigmoid(DiagGGNBase, TanhJacobian):
    pass


class DiagGGNTanh(DiagGGNBase, SigmoidJacobian):
    pass


EXTENSIONS = [
    DiagGGNReLU(),
    DiagGGNSigmoid(),
    DiagGGNTanh(),
]
