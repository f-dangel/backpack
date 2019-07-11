from ...derivatives.relu import ReLUDerivatives
from ...derivatives.tanh import TanhDerivatives
from ...derivatives.sigmoid import SigmoidDerivatives
from .cmpbase import CMPBase


class CMPReLU(CMPBase, ReLUDerivatives):
    pass


class CMPSigmoid(CMPBase, TanhDerivatives):
    pass


class CMPTanh(CMPBase, SigmoidDerivatives):
    pass


EXTENSIONS = [
    CMPReLU(),
    CMPSigmoid(),
    CMPTanh(),
]
