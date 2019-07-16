from ...core.derivatives.avgpool2d import AvgPool2DDerivatives
from ...core.derivatives.maxpool2d import MaxPool2DDerivatives
from .hbpbase import HBPBase


class HBPAvgPool2d(HBPBase, AvgPool2DDerivatives):
    pass


class HBPMaxpool2d(HBPBase, MaxPool2DDerivatives):
    pass


EXTENSIONS = [
    HBPMaxpool2d(),
    HBPAvgPool2d(),
]
