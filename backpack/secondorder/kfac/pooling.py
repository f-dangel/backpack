from ...core.derivatives.avgpool2d import AvgPool2DDerivatives
from ...core.derivatives.maxpool2d import MaxPool2DDerivatives
from .kfacbase import KFACBase


class KFACAvgPool2d(KFACBase, AvgPool2DDerivatives):
    pass


class KFACMaxpool2d(KFACBase, MaxPool2DDerivatives):
    pass


EXTENSIONS = [
    KFACMaxpool2d(),
    KFACAvgPool2d(),
]
