"""SGS extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase
from backpack.utils.errors import batch_norm_raise_error_if_train


class SGSBatchNormNd(SGSBase):
    """SGS extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"])

    def __call__(self, ext, module, g_inp, g_out):  # noqa: D102
        batch_norm_raise_error_if_train(module)
        super().__call__(ext, module, g_inp, g_out)
