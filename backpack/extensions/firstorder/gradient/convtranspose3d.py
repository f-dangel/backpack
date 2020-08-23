from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives

from .base import GradBaseModule


class GradConvTranspose3d(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(), params=["bias", "weight"]
        )
