from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPDropout(HMPBase):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
