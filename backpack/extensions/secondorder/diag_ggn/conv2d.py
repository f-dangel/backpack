from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.diag_ggn.conv_base import DiagGGNConvBase
from backpack.utils import conv as convUtils


class DiagGGNConv2d(DiagGGNConvBase):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(), N=2, params=["bias", "weight"]
        )
