from ..base.tanh import BaseTanh
from .elementwise import DiagGGNElementwise


class DiagGGNTanh(DiagGGNElementwise, BaseTanh):
    pass


EXTENSIONS = [DiagGGNTanh()]
