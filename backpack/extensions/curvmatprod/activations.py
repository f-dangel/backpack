from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from .cmpbase import CMPBase


class CMPReLU(CMPBase):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class CMPSigmoid(CMPBase):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class CMPTanh(CMPBase):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
