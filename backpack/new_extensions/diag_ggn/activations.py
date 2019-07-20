from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNReLU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class DiagGGNSigmoid(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class DiagGGNTanh(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
