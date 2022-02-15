"""Module extensions for custom properties of HBPBaseModule."""
from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPScaleModule(HBPBaseModule):
    """HBP extension for ScaleModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())


class HBPSumModule(HBPBaseModule):
    """HBP extension for SumModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=SumModuleDerivatives())
