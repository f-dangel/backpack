from ..base.sigmoid import BaseSigmoid
from .elementwise import DiagGGNElementwise


class DiagGGNSigmoid(DiagGGNElementwise, BaseSigmoid):
    pass


EXTENSIONS = [DiagGGNSigmoid()]
