"""Module implementing GGN for RNN."""
from backpack.core.derivatives.lstm import LSTMDerivatives
from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNRNN(DiagGGNBaseModule):
    """Calculating diagonal of GGN."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
            sum_batch=True,
        )


class BatchDiagGGNRNN(DiagGGNBaseModule):
    """Calculating per-sample diagonal of GGN."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
            sum_batch=False,
        )


class DiagGGNLSTM(DiagGGNBaseModule):
    """Calculating GGN diagonal of LSTM."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=LSTMDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
            sum_batch=True,
        )


class BatchDiagGGNLSTM(DiagGGNBaseModule):
    """Calculating per-sample diagonal of GGN."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=LSTMDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
            sum_batch=False,
        )
