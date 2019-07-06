from ...jacobians.avgpool2d import AvgPool2dJacobian
from ...jacobians.maxpool2d import MaxPool2dJacobian
from .kflrbase import KFLRBase


class KFLRAvgPool2d(KFLRBase, AvgPool2dJacobian):
    pass


class KFLRMaxpool2d(KFLRBase, MaxPool2dJacobian):
    pass


EXTENSIONS = [
    KFLRMaxpool2d(),
    KFLRAvgPool2d(),
]
