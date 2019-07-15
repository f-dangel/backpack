from ...core.derivatives.dropout import DropoutDerivatives
from .hbpbase import HBPBase


class HBPDropout(HBPBase, DropoutDerivatives):
    pass


EXTENSIONS = [HBPDropout()]
