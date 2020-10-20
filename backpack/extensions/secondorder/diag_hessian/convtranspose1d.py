from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.secondorder.diag_hessian.conv_base import DiagHConvBase


class DiagHConvTranspose1d(DiagHConvBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
            convtranspose=True,
        )
