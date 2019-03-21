"""Hessian backpropagation for the view operation."""

from .reshape import HBPReshape


class HBPView(HBPReshape):
    """The HBP for both view and reshape operation of a tensor are the same."""

    def forward(self, input):
        """Apply the view operation."""
        return input.view(*self._target_shape)


class HBPViewBatchFlat(HBPView):
    """Keep dimension 1, flatten dimensions 2, 3, ... into one dimension."""

    def __init__(self):
        super().__init__(shape=None)

    def forward(self, input):
        """Apply the matricization along dimensions 2, 3, ... ."""
        new_shape = (input.size()[0], -1)
        return input.view(*new_shape)
