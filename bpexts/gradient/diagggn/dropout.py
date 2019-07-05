from ..jacobians.dropout import DropoutJacobian
from .base import DiagGGNBase


class DiagGGNDropout(DiagGGNBase, DropoutJacobian):
    pass


EXTENSIONS = [DiagGGNDropout()]
