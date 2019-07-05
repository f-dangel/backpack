from ..jacobians.relu import ReLUJacobian
from ..jacobians.tanh import TanhJacobian
from ..jacobians.sigmoid import SigmoidJacobian
from .elementwise import DiagGGNElementwise


class DiagGGNReLU(DiagGGNElementwise, ReLUJacobian):
    pass


class DiagGGNSigmoid(DiagGGNElementwise, TanhJacobian):
    pass


class DiagGGNTanh(DiagGGNElementwise, SigmoidJacobian):
    pass


EXTENSIONS = [
    DiagGGNReLU(),
    DiagGGNSigmoid(),
    DiagGGNTanh(),
]
