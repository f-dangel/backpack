from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.secondorder.diag_hessian.convnd import (
    BatchDiagHConvND,
    DiagHConvND,
)


class DiagHConv3d(DiagHConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )


class BatchDiagHConv3d(BatchDiagHConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
