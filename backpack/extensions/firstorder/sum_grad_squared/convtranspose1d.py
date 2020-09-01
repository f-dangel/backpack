from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sum_grad_base import SumGradBase


class SGSConvTranspose1d(SumGradBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives, params=["bias", "weight"]
        )
