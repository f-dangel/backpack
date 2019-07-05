from ..jacobians.maxpool2d import MaxPool2dJacobian
from .base import DiagGGNBase


class DiagGGNMaxpool2d(DiagGGNBase, MaxPool2dJacobian):
    pass


EXTENSIONS = [DiagGGNMaxpool2d()]
