"""DiagGGN extension for BatchNorm."""
from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNBatchNormNd(DiagGGNBaseModule):
    """DiagGGN extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"], sum_batch=True)


class BatchDiagGGNBatchNormNd(DiagGGNBaseModule):
    """BatchDiagGGN extension for BatchNorm."""

    def __init__(self):
        """Initialization."""
        super().__init__(BatchNormNdDerivatives(), ["weight", "bias"], sum_batch=False)
