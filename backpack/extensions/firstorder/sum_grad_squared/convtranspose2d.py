from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sum_grad_base import SumGradBase


class SGSConvTranspose2d(SumGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives, params=["bias", "weight"]
        )
