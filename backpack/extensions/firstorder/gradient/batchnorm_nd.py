"""Gradient extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives

from .base import GradBaseModule


class GradBatchNormNd(GradBaseModule):
    """Gradient extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["bias", "weight"]
        )
