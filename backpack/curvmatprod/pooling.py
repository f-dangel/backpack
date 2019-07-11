from ..core.derivatives.avgpool2d import AvgPool2DDerivatives
from ..core.derivatives.maxpool2d import MaxPool2DDerivatives
from .cmpbase import CMPBase


class CMPAvgPool2d(CMPBase, AvgPool2DDerivatives):
    pass


class CMPMaxpool2d(CMPBase, MaxPool2DDerivatives):
    pass


EXTENSIONS = [
    CMPMaxpool2d(),
    CMPAvgPool2d(),
]
