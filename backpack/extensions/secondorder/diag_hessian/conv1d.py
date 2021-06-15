"""Module extensions for diagonal Hessian properties of ``torch.nn.Conv1d``."""
from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.extensions.secondorder.diag_hessian.convnd import (
    BatchDiagHConvND,
    DiagHConvND,
)


class DiagHConv1d(DiagHConvND):
    """Module extension for the Hessian diagonal of ``torch.nn.Conv1d``."""

    def __init__(self):
        """Store parameter names, convolution dimension, and derivatives object."""
        super().__init__(
            derivatives=Conv1DDerivatives(), N=1, params=["bias", "weight"]
        )


class BatchDiagHConv1d(BatchDiagHConvND):
    """Module extension for the per-sample Hessian diagonal of ``torch.nn.Conv1d``."""

    def __init__(self):
        """Store parameter names, convolution dimension, and derivatives object."""
        super().__init__(
            derivatives=Conv1DDerivatives(), N=1, params=["bias", "weight"]
        )
