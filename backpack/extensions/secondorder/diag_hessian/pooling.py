from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.secondorder.diag_hessian.diag_h_base import DiagHBaseModule


class DiagHAvgPool2d(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class DiagHMaxPool2d(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())
