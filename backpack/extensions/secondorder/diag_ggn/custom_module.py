from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNScaleModule(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ScaleModuleDerivatives())


class BatchDiagGGNScaleModule(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ScaleModuleDerivatives())


class DiagGGNSumModule(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SumModuleDerivatives())


class BatchDiagGGNSumModule(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SumModuleDerivatives())
