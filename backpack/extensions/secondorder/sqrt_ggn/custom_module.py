"""``SqrtGGN{Exact, MC}`` extensions for BackPACK's custom modules."""

from backpack.core.derivatives.scale_module import ScaleModuleDerivatives
from backpack.core.derivatives.sum_module import SumModuleDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNScaleModule(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``ScaleModule``."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=ScaleModuleDerivatives())


class SqrtGGNSumModule(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``SumModule``."""

    def __init__(self):
        """Initialization."""
        super().__init__(derivatives=SumModuleDerivatives())
