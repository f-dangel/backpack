"""Contains BatchGradRNN."""
from backpack.core.derivatives.lstm import LSTMDerivatives
from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.extensions.firstorder.batch_grad.batch_grad_base import BatchGradBase


class BatchGradRNN(BatchGradBase):
    """Extension for RNN calculating grad_batch."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )


class BatchGradLSTM(BatchGradBase):
    """Extension for LSTM calculating grad_batch."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            derivatives=LSTMDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )
