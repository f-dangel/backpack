"""Contains extensions for pooling layers used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.avgpool1d import AvgPool1DDerivatives
from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.avgpool3d import AvgPool3DDerivatives
from backpack.core.derivatives.maxpool1d import MaxPool1DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.core.derivatives.maxpool3d import MaxPool3DDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNMaxPool1d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.MaxPool1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.MaxPool1d`` module."""
        super().__init__(MaxPool1DDerivatives())


class SqrtGGNMaxPool2d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.MaxPool2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.MaxPool2d`` module."""
        super().__init__(MaxPool2DDerivatives())


class SqrtGGNMaxPool3d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.MaxPool3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.MaxPool3d`` module."""
        super().__init__(MaxPool3DDerivatives())


class SqrtGGNAvgPool1d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.AvgPool1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.AvgPool1d`` module."""
        super().__init__(AvgPool1DDerivatives())


class SqrtGGNAvgPool2d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.AvgPool2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.AvgPool2d`` module."""
        super().__init__(AvgPool2DDerivatives())


class SqrtGGNAvgPool3d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.AvgPool3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.AvgPool3d`` module."""
        super().__init__(AvgPool3DDerivatives())
