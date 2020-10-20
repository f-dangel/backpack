from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.secondorder.diag_ggn.conv_base import DiagGGNConvBase
from backpack.utils import conv as convUtils


class DiagGGNConv3d(DiagGGNConvBase):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
            convtranspose=False,
        )
