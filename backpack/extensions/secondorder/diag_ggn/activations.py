from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNReLU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class DiagGGNSigmoid(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class DiagGGNTanh(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())


class DiagGGNELU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ELUDerivatives())


class DiagGGNSELU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SELUDerivatives())


class DiagGGNLeakyReLU(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LeakyReLUDerivatives())


class DiagGGNLogSigmoid(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LogSigmoidDerivatives())
