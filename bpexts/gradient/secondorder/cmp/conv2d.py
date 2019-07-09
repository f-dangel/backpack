import torch
from ...utils import conv as convUtils
from ...derivatives.conv2d import Conv2DDerivatives
from ....utils import einsum
from .cmpbase import CMPBase


class CMPConv2d(CMPBase, Conv2DDerivatives):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, module, grad_input, grad_output):
        raise NotImplementedError

    def bias(self, module, grad_input, grad_output):
        raise NotImplementedError


EXTENSIONS = [CMPConv2d()]
