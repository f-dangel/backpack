from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.secondorder.diag_hessian.conv_base import DiagHConvBase


class DiagHConvTranspose3d(DiagHConvBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
            convtranspose=True,
        )
