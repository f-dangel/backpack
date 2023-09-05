"""DiagH extensions for backpack's custom modules."""
from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHScaleModule(DiagHBaseModule):
    """DiagH extension for ScaleModule."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())
