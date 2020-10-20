from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.secondorder.diag_ggn.conv_base import DiagGGNConvBase
from backpack.utils import conv as convUtils


class DiagGGNConv1d(DiagGGNConvBase):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
            convtranspose=False,
        )
