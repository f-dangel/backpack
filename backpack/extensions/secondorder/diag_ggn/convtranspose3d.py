from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.secondorder.diag_ggn.convtransposend import (
    BatchDiagGGNConvTransposeND,
    DiagGGNConvTransposeND,
)


class DiagGGNConvTranspose3d(DiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(), params=["bias", "weight"]
        )


class BatchDiagGGNConvTranspose3d(BatchDiagGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(), params=["bias", "weight"]
        )
