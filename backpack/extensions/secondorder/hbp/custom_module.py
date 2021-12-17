from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPSumModule(HBPBaseModule):
    """KFAC extension for SumModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=SumModuleDerivatives())
