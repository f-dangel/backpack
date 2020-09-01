from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sum_grad_base import SumGradBase


class SGSConvTranspose3d(SumGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives, params=["bias", "weight"]
        )
