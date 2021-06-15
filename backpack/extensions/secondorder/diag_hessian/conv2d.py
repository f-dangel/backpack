"""Module extensions for diagonal Hessian properties of ``torch.nn.Conv2d``."""
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.diag_hessian.convnd import (
    BatchDiagHConvND,
    DiagHConvND,
)


class DiagHConv2d(DiagHConvND):
    """Module extension for the Hessian diagonal of ``torch.nn.Conv2d``."""

    def __init__(self):
        """Store parameter names and derivatives object."""
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])


class BatchDiagHConv2d(BatchDiagHConvND):
    """Module extension for the per-sample Hessian diagonal of ``torch.nn.Conv2d``."""

    def __init__(self):
        """Store parameter names and derivatives object."""
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
