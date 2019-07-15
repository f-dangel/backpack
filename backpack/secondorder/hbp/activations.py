from ...core.derivatives.relu import ReLUDerivatives
from ...core.derivatives.tanh import TanhDerivatives
from ...core.derivatives.sigmoid import SigmoidDerivatives
from .hbpbase import HBPBase


class HBPReLU(HBPBase, ReLUDerivatives):
    pass


class HBPSigmoid(HBPBase, TanhDerivatives):
    pass


class HBPTanh(HBPBase, SigmoidDerivatives):
    pass


EXTENSIONS = [
    HBPReLU(),
    HBPSigmoid(),
    HBPTanh(),
]
