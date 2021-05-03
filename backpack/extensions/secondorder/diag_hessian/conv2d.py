from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.diag_hessian.convnd import (
    BatchDiagHConvND,
    DiagHConvND,
)


class DiagHConv2d(DiagHConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )


class BatchDiagHConv2d(BatchDiagHConvND):
    '''
    Individual Diagonal of the Hessian for torch.nn.Conv2d
    '''
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
