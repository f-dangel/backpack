from ..base.relu import BaseReLU
from ..base.tanh import BaseTanh
from ..base.sigmoid import BaseSigmoid
from .elementwise import DiagGGNElementwise


class DiagGGNReLU(DiagGGNElementwise, BaseReLU):
    pass


class DiagGGNSigmoid(DiagGGNElementwise, BaseSigmoid):
    pass


class DiagGGNTanh(DiagGGNElementwise, BaseTanh):
    pass


EXTENSIONS = [
    DiagGGNReLU(),
    DiagGGNSigmoid(),
    DiagGGNTanh(),
]
