"""Contains SGSRNN module."""
from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSRNN(SGSBase):
    """Extension for RNN, calculating sum_gradient_squared."""

    def __init__(self):
        """Initialization."""
        super(SGSRNN, self).__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )
