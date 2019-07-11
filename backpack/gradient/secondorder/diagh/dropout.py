from ....core.derivatives.dropout import DropoutDerivatives
from .diaghbase import DiagHBase

DETACH_INPUTS = True


class DiagHDropout(DiagHBase, DropoutDerivatives):
    pass


EXTENSIONS = [DiagHDropout()]
