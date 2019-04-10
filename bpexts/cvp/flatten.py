"""Curvature-vector products of batch-wise flatten operation."""

from .view import CVPView


class CVPFlatten(CVPView):
    """Keep dimension 1, flatten dimensions 2, 3, ... into one dimension."""

    def __init__(self):
        super().__init__(shape=None)

    def forward(self, input):
        """Apply the matricization along dimensions 2, 3, ... ."""
        new_shape = (input.size()[0], -1)
        return input.view(*new_shape)
