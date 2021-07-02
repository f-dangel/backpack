"""Contains transpose convolution layer extensions used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNConvTranspose1d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.ConvTranspose1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose1d`` module."""
        super().__init__(ConvTranspose1DDerivatives(), params=["bias", "weight"])


class SqrtGGNConvTranspose2d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.ConvTranspose2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose2d`` module."""
        super().__init__(ConvTranspose2DDerivatives(), params=["bias", "weight"])


class SqrtGGNConvTranspose3d(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.ConvTranspose3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose3d`` module."""
        super().__init__(ConvTranspose3DDerivatives(), params=["bias", "weight"])
