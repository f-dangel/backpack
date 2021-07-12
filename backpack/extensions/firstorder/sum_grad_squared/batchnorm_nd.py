"""SGS extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSBatchNormNd(SGSBase):
    """SGS extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"])
