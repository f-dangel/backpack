from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import (
    HEAD,
    BatchDiagGGNConvND,
    DiagGGNConvND,
    6e2f6ace71d1aac118f878f968753ac9e83f742d,
    <<<<<<<,
    =======,
    >>>>>>>,
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
