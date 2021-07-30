"""DiagGGN extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils.errors import batch_norm_raise_error_if_train


class DiagGGNBatchNormNd(DiagGGNBaseModule):
    """DiagGGN extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"], sum_batch=True)

    def __call__(self, ext, module, g_inp, g_out):
        batch_norm_raise_error_if_train(module)
        super().__call__(ext, module, g_inp, g_out)


class BatchDiagGGNBatchNormNd(DiagGGNBaseModule):
    """BatchDiagGGN extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"], sum_batch=False)

    def __call__(self, ext, module, g_inp, g_out):
        batch_norm_raise_error_if_train(module)
        super().__call__(ext, module, g_inp, g_out)
