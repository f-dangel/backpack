from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.secondorder.diag_ggn.convtransposend import (
    DiagGGNConvTransposeND,
    BatchDiagGGNConvTransposeND,
)


class DiagGGNConvTranspose1d(DiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )


class BatchDiagGGNConvTranspose1d(BatchDiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
