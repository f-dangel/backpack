from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.secondorder.diag_ggn.convnd import DiagGGNConvND, BatchDiagGGNConvND


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
