from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sum_grad_base import SumGradBase


class SGSConv2d(SumGradBase):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives, params=["bias", "weight"])
