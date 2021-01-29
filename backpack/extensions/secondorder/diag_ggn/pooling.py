from backpack.core.derivatives.avgpool1d import AvgPool1DDerivatives
from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.avgpool3d import AvgPool3DDerivatives
from backpack.core.derivatives.maxpool1d import MaxPool1DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.core.derivatives.maxpool3d import MaxPool3DDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNMaxPool1d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool1DDerivatives())


class DiagGGNMaxPool2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class DiagGGNAvgPool1d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool1DDerivatives())


class DiagGGNMaxPool3d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool3DDerivatives())


class DiagGGNAvgPool2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class DiagGGNAvgPool3d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool3DDerivatives())
