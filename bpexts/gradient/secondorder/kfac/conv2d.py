from ...utils import conv as convUtils
from ...derivatives.conv2d import Conv2DDerivatives
from ....utils import einsum
from .kfacbase import KFACBase


class KFACConv2d(KFACBase, Conv2DDerivatives):
    def __init__(self):
        raise NotImplementedError
