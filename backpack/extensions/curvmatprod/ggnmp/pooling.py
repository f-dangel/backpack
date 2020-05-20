from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPAvgPool2d(GGNMPBase, AvgPool2DDerivatives):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class GGNMPMaxpool2d(GGNMPBase, MaxPool2DDerivatives):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())
