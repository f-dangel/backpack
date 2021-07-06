from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradBatchNorm1d(BatchGradBase):
    def __init__(self):
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["bias", "weight"]
        )
