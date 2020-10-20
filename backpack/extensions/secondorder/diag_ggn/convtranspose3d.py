from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.secondorder.diag_ggn.conv_base import DiagGGNConvBase


class DiagGGNConvTranspose3d(DiagGGNConvBase):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
            convtranspose=True,
        )
