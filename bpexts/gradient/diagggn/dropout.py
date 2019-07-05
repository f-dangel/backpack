from ..base.dropout import BaseDropout
from .elementwise import DiagGGNElementwise


class DiagGGNDropout(DiagGGNElementwise, BaseDropout):
    pass


EXTENSIONS = [DiagGGNDropout()]
