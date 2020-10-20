from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.secondorder.diag_hessian.conv_base import DiagHConvBase


class DiagHConv3d(DiagHConvBase):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(), N=3, params=["bias", "weight"]
        )
