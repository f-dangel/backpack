"""Variance extension for BatchNorm."""
from backpack.extensions.firstorder.gradient.batchnorm_nd import GradBatchNormNd
from backpack.extensions.firstorder.sum_grad_squared.batchnorm_nd import SGSBatchNorm
from backpack.extensions.firstorder.variance.variance_base import VarianceBaseModule


class VarianceBatchNorm(VarianceBaseModule):
    """Variance extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], GradBatchNormNd(), SGSBatchNorm())
