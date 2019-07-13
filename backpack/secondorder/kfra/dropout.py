from ...core.derivatives.dropout import DropoutDerivatives
from .kfrabase import KFRABase


class KFRADropout(KFRABase, DropoutDerivatives):
    pass


EXTENSIONS = [KFRADropout()]
