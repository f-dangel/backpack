"""Contains grad_batch extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase
from backpack.utils.errors import batch_norm_raise_error_if_train


class BatchGradBatchNormNd(BatchGradBase):
    """BatchGrad extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["bias", "weight"]
        )

    def __call__(self, ext, module, g_inp, g_out):
        batch_norm_raise_error_if_train(module, raise_error=False)
        super().__call__(ext, module, g_inp, g_out)
