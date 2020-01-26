from backpack.core.derivatives.dropout import DropoutDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPDropout(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
