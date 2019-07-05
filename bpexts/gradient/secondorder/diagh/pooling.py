from ...jacobians.avgpool2d import AvgPool2dJacobian
from ...jacobians.maxpool2d import MaxPool2dJacobian
from .diaghbase import DiagHBase


class DiagHAvgPool2d(DiagHBase, AvgPool2dJacobian):
    pass


class DiagHMaxPool2d(DiagHBase, MaxPool2dJacobian):
    pass


EXTENSIONS = [
    DiagHAvgPool2d(),
    DiagHMaxPool2d(),
]
