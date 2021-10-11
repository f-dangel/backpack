"""Contains GradRNN."""
from backpack.core.derivatives.lstm import LSTMDerivatives
from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.extensions.firstorder.gradient.base import GradBaseModule


class GradRNN(GradBaseModule):
    """Extension for RNN, calculating gradient."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )


class GradLSTM(GradBaseModule):
    """Extension for LSTM, calculating gradient."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=LSTMDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )
