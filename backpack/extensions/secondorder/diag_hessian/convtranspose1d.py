from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.extensions.secondorder.diag_hessian.convtransposend import (
    BatchDiagHConvTransposeND,
    DiagHConvTransposeND,
)


class DiagHConvTranspose1d(DiagHConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )


class BatchDiagHConvTranspose1d(BatchDiagHConvTransposeND):
    '''
    Individual Diagonal of the Hessian for torch.nn.ConvTranspose1d
    '''
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
