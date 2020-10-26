from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSConvTranspose3d(SGSBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(), params=["bias", "weight"]
        )
