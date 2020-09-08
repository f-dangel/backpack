from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSConvTranspose2d(SGSBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(), params=["bias", "weight"]
        )
