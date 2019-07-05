from .base import DiagGGNBase
from ..jacobians.avgpool2d import AvgPool2dJacobian


class DiagGGNAvgPool2d(DiagGGNBase, AvgPool2dJacobian):
    pass


EXTENSIONS = [DiagGGNAvgPool2d()]
