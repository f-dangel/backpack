from ....core.derivatives.relu import ReLUDerivatives
from ....core.derivatives.sigmoid import SigmoidDerivatives
from ....core.derivatives.tanh import TanhDerivatives
from .diaghbase import DiagHBase


class DiagHReLU(DiagHBase, ReLUDerivatives):
    pass


class DiagHSigmoid(DiagHBase, SigmoidDerivatives):
    pass


class DiagHTanh(DiagHBase, TanhDerivatives):
    pass


EXTENSIONS = [
    DiagHReLU(),
    DiagHSigmoid(),
    DiagHTanh(),
]
