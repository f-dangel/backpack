"""Contains extensions for activation layers used by ``GGNMP``."""
from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPReLU(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.ReLU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ReLU`` module."""
        super().__init__(ReLUDerivatives())


class GGNMPSigmoid(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.Sigmoid`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Sigmoid`` module."""
        super().__init__(SigmoidDerivatives())


class GGNMPTanh(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.Tanh`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Tanh`` module."""
        super().__init__(TanhDerivatives())


class GGNMPELU(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.ELU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ELU`` module."""
        super().__init__(ELUDerivatives())


class GGNMPSELU(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.SELU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.SELU`` module."""
        super().__init__(SELUDerivatives())


class GGNMPLeakyReLU(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.LeakyReLU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.LeakyReLU`` module."""
        super().__init__(LeakyReLUDerivatives())


class GGNMPLogSigmoid(GGNMPBase):
    """``GGNMP`` extension for ``torch.nn.LogSigmoid`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.LogSigmoid`` module."""
        super().__init__(LogSigmoidDerivatives())
