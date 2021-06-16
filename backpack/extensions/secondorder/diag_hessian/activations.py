from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
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


class DiagHELU(DiagHBaseModule):
    """Module extension that computes the Hessian diagonal for ``torch.nn.ELU``."""

    def __init__(self):
        super().__init__(derivatives=ELUDerivatives())


class DiagHSELU(DiagHBaseModule):
    """Module extension that computes the Hessian diagonal for ``torch.nn.SELU``."""

    def __init__(self):
        super().__init__(derivatives=SELUDerivatives())
