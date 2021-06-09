"""Module implementing GGN."""
from backpack.core.derivatives.rnn import RNNDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNRNN(DiagGGNBaseModule):
    """Calculating GGN derivative."""

    def __init__(self):
        """Initialize."""
        super().__init__(
            derivatives=RNNDerivatives(),
            params=["bias_ih_l0", "bias_hh_l0", "weight_ih_l0", "weight_hh_l0"],
        )
