from ...core.derivatives.relu import ReLUDerivatives
from ...core.derivatives.tanh import TanhDerivatives
from ...core.derivatives.sigmoid import SigmoidDerivatives
from .kfacbase import KFACBase


class KFACReLU(KFACBase, ReLUDerivatives):
    pass


class KFACSigmoid(KFACBase, TanhDerivatives):
    pass


class KFACTanh(KFACBase, SigmoidDerivatives):
    pass


EXTENSIONS = [
    KFACReLU(),
    KFACSigmoid(),
    KFACTanh(),
]
