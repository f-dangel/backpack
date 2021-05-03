from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.extensions.secondorder.diag_hessian.convtransposend import (
    BatchDiagHConvTransposeND,
    DiagHConvTransposeND,
)


class DiagHConvTranspose2d(DiagHConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )


class BatchDiagHConvTranspose2d(BatchDiagHConvTransposeND):
    """
    Individual Diagonal of the Hessian for torch.nn.ConvTranspose2d
    """

    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
