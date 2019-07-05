from ...jacobians.dropout import DropoutJacobian
from .diaghbase import DiagHBase

DETACH_INPUTS = True


class DiagHDropout(DiagHBase, DropoutJacobian):
    pass


EXTENSIONS = [DiagHDropout()]
