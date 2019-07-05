from ..jacobians.dropout import DropoutJacobian
from .elementwise import DiagGGNElementwise


class DiagGGNDropout(DiagGGNElementwise, DropoutJacobian):
    pass


EXTENSIONS = [DiagGGNDropout()]
