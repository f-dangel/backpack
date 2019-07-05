from ..base.relu import BaseReLU
from .elementwise import DiagGGNElementwise


class DiagGGNReLU(DiagGGNElementwise, BaseReLU):
    pass


EXTENSIONS = [DiagGGNReLU()]
