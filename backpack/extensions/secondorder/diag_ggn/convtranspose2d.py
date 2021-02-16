from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.secondorder.diag_ggn.convtransposend import (
    DiagGGNConvTransposeND,
    BatchDiagGGNConvTransposeND,
)


class DiagGGNConvTranspose2d(DiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )


class BatchDiagGGNConvTranspose2d(BatchDiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
