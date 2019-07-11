from ....core.derivatives.avgpool2d import AvgPool2DDerivatives
from ....core.derivatives.maxpool2d import MaxPool2DDerivatives
from .diaghbase import DiagHBase


class DiagHAvgPool2d(DiagHBase, AvgPool2DDerivatives):
    pass


class DiagHMaxPool2d(DiagHBase, MaxPool2DDerivatives):
    pass


EXTENSIONS = [
    DiagHAvgPool2d(),
    DiagHMaxPool2d(),
]
