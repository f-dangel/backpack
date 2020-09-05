from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSConvTranspose1d(SGSBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(), params=["bias", "weight"]
        )
