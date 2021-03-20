from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import (
    HEAD,
    BatchDiagGGNConvND,
    DiagGGNConvND,
    6e2f6ace71d1aac118f878f968753ac9e83f742d,
    <<<<<<<,
    =======,
    >>>>>>>,
)


class DiagGGNConv1d(DiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )


class BatchDiagGGNConv1d(BatchDiagGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
