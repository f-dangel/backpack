from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPFlatten(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
