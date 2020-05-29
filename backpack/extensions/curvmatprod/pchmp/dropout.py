from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPDropout(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
