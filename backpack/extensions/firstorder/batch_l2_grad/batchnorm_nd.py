"""Contains batch_l2 extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base
from backpack.utils.errors import batch_norm_raise_error_if_train


class BatchL2BatchNorm(BatchL2Base):
    """batch_l2 extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], BatchNormNdDerivatives())

    def apply(self, ext, module, g_inp, g_out):  # noqa: D102
        batch_norm_raise_error_if_train(module)
        super().apply(ext, module, g_inp, g_out)
