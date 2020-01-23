from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradBatchNorm1d(BatchGradBase):
    def __init__(self):
        super().__init__(
            derivatives=BatchNorm1dDerivatives(), params=["bias", "weight"]
        )
