from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.secondorder.diag_ggn.convtransposend import (
    DiagGGNConvTransposeND,
    BatchDiagGGNConvTransposeND,
)


class DiagGGNConvTranspose3d(DiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )


class BatchDiagGGNConvTranspose3d(BatchDiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
