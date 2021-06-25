"""Contains hbp extension for ScaleModule."""
from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPScaleModule(HBPBaseModule):
    """HBP extension for ScaleModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())
