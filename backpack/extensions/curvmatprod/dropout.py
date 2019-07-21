from backpack.core.derivatives.dropout import DropoutDerivatives
from .cmpbase import CMPBase


class CMPDropout(CMPBase):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
