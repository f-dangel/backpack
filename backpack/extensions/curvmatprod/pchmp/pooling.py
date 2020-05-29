from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPAvgPool2d(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class PCHMPMaxpool2d(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())
