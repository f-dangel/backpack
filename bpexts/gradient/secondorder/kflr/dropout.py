from ...derivatives.dropout import DropoutDerivatives
from .kflrbase import KFLRBase


class KFLRDropout(KFLRBase, DropoutDerivatives):
    pass


EXTENSIONS = [KFLRDropout()]
