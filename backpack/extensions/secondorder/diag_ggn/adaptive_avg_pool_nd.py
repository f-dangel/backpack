"""DiagGGN extension for AdaptiveAvgPool."""
from backpack.core.derivatives.adaptive_avg_pool_nd import AdaptiveAvgPoolNDDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNAdaptiveAvgPoolNd(DiagGGNBaseModule):
    """DiagGGN extension for AdaptiveAvgPool."""

    def __init__(self, N: int):
        """Initialization.

        Args:
            N: number of free dimensions, e.g. use N=1 for AdaptiveAvgPool1d
        """
        super().__init__(derivatives=AdaptiveAvgPoolNDDerivatives(N=N))
