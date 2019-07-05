from ...jacobians.relu import ReLUJacobian
from ...jacobians.tanh import TanhJacobian
from ...jacobians.sigmoid import SigmoidJacobian
from .base import KFLRBase


class KFLRReLU(KFLRBase, ReLUJacobian):
    pass


class KFLRSigmoid(KFLRBase, TanhJacobian):
    pass


class KFLRTanh(KFLRBase, SigmoidJacobian):
    pass


EXTENSIONS = [
    KFLRReLU(),
    KFLRSigmoid(),
    KFLRTanh(),
]
