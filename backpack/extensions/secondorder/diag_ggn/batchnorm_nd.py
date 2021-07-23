"""DiagGGN extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils.errors import batch_norm_raise_error_if_train


class DiagGGNBatchNormNd(DiagGGNBaseModule):
    """DiagGGN extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"], sum_batch=True)

    def apply(self, ext, module, g_inp, g_out):  # noqa: D102
        batch_norm_raise_error_if_train(module)
        super().apply(ext, module, g_inp, g_out)


class BatchDiagGGNBatchNormNd(DiagGGNBaseModule):
    """BatchDiagGGN extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"], sum_batch=False)

    def apply(self, ext, module, g_inp, g_out):  # noqa: D102
        batch_norm_raise_error_if_train(module)
        super().apply(ext, module, g_inp, g_out)
