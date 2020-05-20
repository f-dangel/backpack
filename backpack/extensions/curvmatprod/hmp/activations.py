from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPReLU(HMPBase):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class HMPSigmoid(HMPBase):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class HMPTanh(HMPBase):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())
