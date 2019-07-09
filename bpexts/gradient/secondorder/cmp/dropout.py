from ...derivatives.dropout import DropoutDerivatives
from .cmpbase import CMPBase


class CMPDropout(CMPBase, DropoutDerivatives):
    pass


EXTENSIONS = [CMPDropout()]
