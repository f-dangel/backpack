from ...core.derivatives.dropout import DropoutDerivatives
from .diagggnbase import DiagGGNBase


class DiagGGNDropout(DiagGGNBase, DropoutDerivatives):
    pass


EXTENSIONS = [DiagGGNDropout()]
