"""Hessian backpropagation for the view operation."""

from .reshape import HBPReshape


class HBPView(HBPReshape):
    """The HBP for both view and reshape operation of a tensor are the same."""

    def forward(self, input):
        """Apply the transposition operation."""
        return input.view(*self._target_shape)
