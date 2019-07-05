from ...jacobians.dropout import DropoutJacobian
from .diagggnbase import DiagGGNBase


class DiagGGNDropout(DiagGGNBase, DropoutJacobian):
    pass


EXTENSIONS = [DiagGGNDropout()]
