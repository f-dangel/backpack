from ...core.derivatives.avgpool2d import AvgPool2DDerivatives
from ...core.derivatives.maxpool2d import MaxPool2DDerivatives
from .diagggnbase import DiagGGNBase


class DiagGGNAvgPool2d(DiagGGNBase, AvgPool2DDerivatives):
    pass


class DiagGGNMaxpool2d(DiagGGNBase, MaxPool2DDerivatives):
    pass


EXTENSIONS = [
    DiagGGNMaxpool2d(),
    DiagGGNAvgPool2d(),
]
