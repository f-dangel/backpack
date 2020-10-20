from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.secondorder.diag_ggn.conv_base import DiagGGNConvBase


class DiagGGNConvTranspose1d(DiagGGNConvBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
            convtranspose=True,
        )
