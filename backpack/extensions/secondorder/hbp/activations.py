from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPReLU(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class HBPSigmoid(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class HBPTanh(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
