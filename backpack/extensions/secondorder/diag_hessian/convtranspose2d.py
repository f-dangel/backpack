from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.secondorder.diag_hessian.conv_base import DiagHConvBase


class DiagHConvTranspose2d(DiagHConvBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
            convtranspose=True,
        )
