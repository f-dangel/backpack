from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule
from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHReLU(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class DiagHSigmoid(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class DiagHTanh(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())


class DiagGGNELU(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=ELUDerivatives())


class DiagGGNSELU(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=SELUDerivatives())


class DiagGGNLeakyReLU(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LeakyReLUDerivatives())


class DiagGGNLogSigmoid(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LogSigmoidDerivatives())
