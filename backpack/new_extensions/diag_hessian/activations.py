from ...core.derivatives.relu import ReLUDerivatives
from ...core.derivatives.sigmoid import SigmoidDerivatives
from ...core.derivatives.tanh import TanhDerivatives
from .diag_h_base import DiagHBaseModule


class DiagHReLU(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class DiagHSigmoid(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class DiagHTanh(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
