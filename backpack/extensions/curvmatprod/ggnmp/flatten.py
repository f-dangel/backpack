from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPFlatten(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
