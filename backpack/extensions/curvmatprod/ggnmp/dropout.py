from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPDropout(GGNMPBase):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
