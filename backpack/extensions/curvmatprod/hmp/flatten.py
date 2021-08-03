from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPFlatten(HMPBase):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
