"""Variance extension for BatchNorm."""
from backpack.extensions.firstorder.gradient.batchnorm_nd import GradBatchNormNd
from backpack.extensions.firstorder.sum_grad_squared.batchnorm_nd import SGSBatchNormNd
from backpack.extensions.firstorder.variance.variance_base import VarianceBaseModule
from backpack.utils.errors import batch_norm_raise_error_if_train


class VarianceBatchNormNd(VarianceBaseModule):
    """Variance extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(["weight", "bias"], GradBatchNormNd(), SGSBatchNormNd())

    def apply(self, ext, module, g_inp, g_out):  # noqa: D102
        batch_norm_raise_error_if_train(module)
        super().apply(ext, module, g_inp, g_out)
