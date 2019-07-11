from ...derivatives.dropout import DropoutDerivatives
from .kfacbase import KFACBase


class KFACDropout(KFACBase, DropoutDerivatives):
    pass


EXTENSIONS = [KFACDropout()]
