"""Gradient extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.utils.errors import batch_norm_raise_error_if_train

from .base import GradBaseModule


class GradBatchNormNd(GradBaseModule):
    """Gradient extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=BatchNormNdDerivatives(), params=["bias", "weight"]
        )

    def __call__(self, ext, module, g_inp, g_out):  # noqa: D102
        batch_norm_raise_error_if_train(module)
        super().__call__(ext, module, g_inp, g_out)
