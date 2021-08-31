"""Contains VarianceRNN."""
from torch.nn import LSTM, RNN

from backpack.extensions.firstorder.gradient.rnn import GradLSTM, GradRNN
from backpack.extensions.firstorder.sum_grad_squared.rnn import SGSLSTM, SGSRNN
from backpack.extensions.firstorder.variance.variance_base import VarianceBaseModule


class VarianceRNN(VarianceBaseModule):
    """Extension for RNN, calculating variance."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
            grad_extension=GradRNN(),
            sgs_extension=SGSRNN(),
        )

    @staticmethod
    def _get_axis_batch(module: RNN) -> int:
        return 0 if module.batch_first else 1


class VarianceLSTM(VarianceBaseModule):
    """Extension for LSTM, calculating variance."""

    def __init__(self):
        """Initialization."""
        super().__init__(
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
            grad_extension=GradLSTM(),
            sgs_extension=SGSLSTM(),
        )

    @staticmethod
    def _get_axis_batch(module: LSTM) -> int:
        return 0 if module.batch_first else 1
