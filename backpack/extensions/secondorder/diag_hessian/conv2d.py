from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.diag_hessian.conv_base import DiagHConvBase


class DiagHConv2d(DiagHConvBase):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(), N=2, params=["bias", "weight"]
        )
