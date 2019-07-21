from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from .cmpbase import CMPBase


class CMPAvgPool2d(CMPBase, AvgPool2DDerivatives):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class CMPMaxpool2d(CMPBase, MaxPool2DDerivatives):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())

