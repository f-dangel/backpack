from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSConv2d(SGSBase):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
