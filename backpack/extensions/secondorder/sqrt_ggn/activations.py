"""Contains extensions for activation layers used by ``SqrtGGN{Exact, MC}``."""
from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from backpack.extensions.secondorder.sqrt_ggn.base import SqrtGGNBaseModule


class SqrtGGNReLU(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.ReLU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ReLU`` module."""
        super().__init__(ReLUDerivatives())


class SqrtGGNSigmoid(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Sigmoid`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Sigmoid`` module."""
        super().__init__(SigmoidDerivatives())


class SqrtGGNTanh(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.Tanh`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Tanh`` module."""
        super().__init__(TanhDerivatives())


class SqrtGGNELU(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.ELU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ELU`` module."""
        super().__init__(ELUDerivatives())


class SqrtGGNSELU(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.SELU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.SELU`` module."""
        super().__init__(SELUDerivatives())


class SqrtGGNLeakyReLU(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.LeakyReLU`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.LeakyReLU`` module."""
        super().__init__(LeakyReLUDerivatives())


class SqrtGGNLogSigmoid(SqrtGGNBaseModule):
    """``SqrtGGN{Exact, MC}`` extension for ``torch.nn.LogSigmoid`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.LogSigmoid`` module."""
        super().__init__(LogSigmoidDerivatives())
