"""Curvature-vector products of the view operation."""

from .reshape import CVPReshape


class CVPView(CVPReshape):
    """View layer with recursive Hessian-vector products.

    The CVP for view and reshape operation are the same."""

    def forward(self, input):
        """Apply the view operation."""
        return input.view(*self._target_shape)
