from backpack.core.derivatives.flatten import FlattenDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPFlatten(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=FlattenDerivatives())
