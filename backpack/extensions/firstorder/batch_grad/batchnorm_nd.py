"""Contains grad_batch extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradBatchNormNd(BatchGradBase):
    """BatchGrad extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["bias", "weight"]
        )
