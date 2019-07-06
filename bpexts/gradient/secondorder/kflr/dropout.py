from ...derivatives.dropout import DropoutDerivatives
from .base import KFLRBase


class KFLRDropout(KFLRBase, DropoutDerivatives):
    pass


EXTENSIONS = [KFLRDropout()]
