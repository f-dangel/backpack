from backpack.core.derivatives.rnn import RNNDerivatives

from .base import GradBaseModule


class GradRNN(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )
