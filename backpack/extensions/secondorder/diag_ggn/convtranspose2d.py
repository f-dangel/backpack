from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.secondorder.diag_ggn.conv_base import DiagGGNConvBase


class DiagGGNConvTranspose2d(DiagGGNConvBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
            convtranspose=True,
        )
