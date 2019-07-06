from ...jacobians.dropout import DropoutJacobian
from .kflrbase import KFLRBase


class KFLRDropout(KFLRBase, DropoutJacobian):
    pass


EXTENSIONS = [KFLRDropout()]
