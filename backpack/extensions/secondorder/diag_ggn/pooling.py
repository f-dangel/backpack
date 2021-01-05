from backpack.core.derivatives.avgpool1d import AvgPool1DDerivatives
from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.avgpool3d import AvgPool3DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNMaxPool2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class DiagGGNAvgPool1d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool1DDerivatives())


class DiagGGNAvgPool2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class DiagGGNAvgPool3d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool3DDerivatives())
