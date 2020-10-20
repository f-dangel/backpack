from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.secondorder.diag_hessian.conv_base import DiagHConvBase


class DiagHConv1d(DiagHConvBase):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
            convtranspose=False,
        )
