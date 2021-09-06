"""DiagGGN extensions for backpack's custom modules."""
from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNScaleModule(DiagGGNBaseModule):
    """DiagGGN extension for ScaleModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())


class DiagGGNSumModule(DiagGGNBaseModule):
    """DiagGGN extension for SumModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=SumModuleDerivatives())
