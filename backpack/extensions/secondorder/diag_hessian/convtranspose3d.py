from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.secondorder.diag_hessian.convtransposend import (
    BatchDiagHConvTransposeND,
    DiagHConvTransposeND,
)


class DiagHConvTranspose3d(DiagHConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )


class BatchDiagHConvTranspose3d(BatchDiagHConvTransposeND):
    '''
    Individual Diagonal of the Hessian for torch.nn.ConvTranspose3d
    '''
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
