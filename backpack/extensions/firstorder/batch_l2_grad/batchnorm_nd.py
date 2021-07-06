"""Contains batch_l2 extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base


class BatchL2BatchNorm(BatchL2Base):
    """batch_l2 extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], BatchNormNdDerivatives())
