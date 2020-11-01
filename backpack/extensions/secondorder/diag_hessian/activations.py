from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
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


class DiagHLeakyReLU(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LeakyReLUDerivatives())


class DiagHLogSigmoid(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LogSigmoidDerivatives())
