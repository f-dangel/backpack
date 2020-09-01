from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sum_grad_base import SumGradBase


class SGSConv3d(SumGradBase):
    def __init__(self):
        super().__init__(derivatives=Conv3DDerivatives, params=["bias", "weight"])
