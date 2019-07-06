from ...derivatives.relu import ReLUDerivatives
from ...derivatives.tanh import TanhDerivatives
from ...derivatives.sigmoid import SigmoidDerivatives
from .kflrbase import KFLRBase


class KFLRReLU(KFLRBase, ReLUDerivatives):
    pass


class KFLRSigmoid(KFLRBase, TanhDerivatives):
    pass


class KFLRTanh(KFLRBase, SigmoidDerivatives):
    pass


EXTENSIONS = [
    KFLRReLU(),
    KFLRSigmoid(),
    KFLRTanh(),
]
