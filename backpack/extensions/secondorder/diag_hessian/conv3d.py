"""Module extensions for diagonal Hessian properties of ``torch.nn.Conv3d``."""
from backpack.core.derivatives.conv3d import Conv3DDerivatives
from backpack.extensions.secondorder.diag_hessian.convnd import (
    BatchDiagHConvND,
    DiagHConvND,
)


class DiagHConv3d(DiagHConvND):
    """Module extension for the Hessian diagonal of ``torch.nn.Conv3d``."""

    def __init__(self):
        """Store parameter names and derivatives object."""
        super().__init__(derivatives=Conv3DDerivatives(), params=["bias", "weight"])


class BatchDiagHConv3d(BatchDiagHConvND):
    """Module extension for the per-sample Hessian diagonal of ``torch.nn.Conv3d``."""

    def __init__(self):
        """Store parameter names and derivatives object."""
        super().__init__(derivatives=Conv3DDerivatives(), params=["bias", "weight"])
