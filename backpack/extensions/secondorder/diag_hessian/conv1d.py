from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.secondorder.diag_hessian.convnd import (
    DiagHConvND,
    BatchDiagHConvND,
)


class BatchDiagHConv1d(BatchDiagHConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
