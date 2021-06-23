from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNScaleModule(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ScaleModuleDerivatives())


class BatchDiagGGNScaleModule(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ScaleModuleDerivatives())
