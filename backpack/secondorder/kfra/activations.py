from ...core.derivatives.relu import ReLUDerivatives
from ...core.derivatives.tanh import TanhDerivatives
from ...core.derivatives.sigmoid import SigmoidDerivatives
from .kfrabase import KFRABase


class KFRAReLU(KFRABase, ReLUDerivatives):
    pass


class KFRASigmoid(KFRABase, TanhDerivatives):
    pass


class KFRATanh(KFRABase, SigmoidDerivatives):
    pass


EXTENSIONS = [
    KFRAReLU(),
    KFRASigmoid(),
    KFRATanh(),
]
