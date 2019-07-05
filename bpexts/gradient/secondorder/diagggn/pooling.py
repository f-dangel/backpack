from ...jacobians.avgpool2d import AvgPool2dJacobian
from ...jacobians.maxpool2d import MaxPool2dJacobian
from .diagggnbase import DiagGGNBase


class DiagGGNAvgPool2d(DiagGGNBase, AvgPool2dJacobian):
    pass


class DiagGGNMaxpool2d(DiagGGNBase, MaxPool2dJacobian):
    pass


EXTENSIONS = [
    DiagGGNMaxpool2d(),
    DiagGGNAvgPool2d(),
]
