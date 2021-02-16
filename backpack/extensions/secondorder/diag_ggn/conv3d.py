from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import (
    DiagGGNConvND,
    BatchDiagGGNConvND,
)


class DiagGGNConv3d(DiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )


class BatchDiagGGNConv3d(BatchDiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
