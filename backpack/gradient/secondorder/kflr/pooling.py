from ....core.derivatives.avgpool2d import AvgPool2DDerivatives
from ....core.derivatives.maxpool2d import MaxPool2DDerivatives
from .kflrbase import KFLRBase


class KFLRAvgPool2d(KFLRBase, AvgPool2DDerivatives):
    pass


class KFLRMaxpool2d(KFLRBase, MaxPool2DDerivatives):
    pass


EXTENSIONS = [
    KFLRMaxpool2d(),
    KFLRAvgPool2d(),
]
