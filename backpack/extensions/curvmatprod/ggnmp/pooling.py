"""Contains extensions for pooling layers used by ``GGNMP``."""
from backpack.core.derivatives.avgpool1d import AvgPool1DDerivatives
from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.avgpool3d import AvgPool3DDerivatives
from backpack.core.derivatives.maxpool1d import MaxPool1DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.core.derivatives.maxpool3d import MaxPool3DDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPMaxPool1d(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.MaxPool1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.MaxPool1d`` module."""
        super().__init__(MaxPool1DDerivatives())


class GGNMPMaxPool2d(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.MaxPool2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.MaxPool2d`` module."""
        super().__init__(MaxPool2DDerivatives())


class GGNMPMaxPool3d(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.MaxPool3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.MaxPool3d`` module."""
        super().__init__(MaxPool3DDerivatives())


class GGNMPAvgPool1d(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.AvgPool1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.AvgPool1d`` module."""
        super().__init__(AvgPool1DDerivatives())


class GGNMPAvgPool2d(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.AvgPool2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.AvgPool2d`` module."""
        super().__init__(AvgPool2DDerivatives())


class GGNMPAvgPool3d(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.AvgPool3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.AvgPool3d`` module."""
        super().__init__(AvgPool3DDerivatives())
