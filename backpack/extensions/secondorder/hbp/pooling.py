from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPAvgPool2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class HBPMaxpool2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())
