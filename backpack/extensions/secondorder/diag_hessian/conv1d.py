from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.secondorder.diag_hessian.convnd import (
    BatchDiagHConvND,
    DiagHConvND,
)


class DiagHConv1d(DiagHConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )


class BatchDiagHConv1d(BatchDiagHConvND):
    '''
    Individual Diagonal of the Hessian for torch.nn.Conv1d
    '''
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
