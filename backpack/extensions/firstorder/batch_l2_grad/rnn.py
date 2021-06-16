"""Contains BatchL2RNN."""
from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base


class BatchL2RNN(BatchL2Base):
    """Extension for RNN, calculating batch_l2."""

    def __init__(self):
        """Initialization."""
        super(BatchL2RNN, self).__init__(
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
            derivatives=RNNDerivatives(),
        )
