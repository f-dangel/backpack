from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import (
    DiagGGNConvND,
    BatchDiagGGNConvND,
)


class DiagGGNConv2d(DiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )


class BatchDiagGGNConv2d(BatchDiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
