from ...jacobians.relu import ReLUJacobian
from ...jacobians.sigmoid import SigmoidJacobian
from ...jacobians.tanh import TanhJacobian
from .diaghbase import DiagHBase


class DiagHReLU(DiagHBase, ReLUJacobian):
    pass


class DiagHSigmoid(DiagHBase, SigmoidJacobian):
    pass


class DiagHTanh(DiagHBase, TanhJacobian):
    pass


EXTENSIONS = [
    DiagHReLU(),
    DiagHSigmoid(),
    DiagHTanh(),
]
