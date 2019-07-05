from ...jacobians.dropout import DropoutJacobian
from .base import KFLRBase


class KFLRDropout(KFLRBase, DropoutJacobian):
    pass


EXTENSIONS = [KFLRDropout()]
