from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives

from .base import GradBaseModule


class GradConvTranspose2d(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(), params=["bias", "weight"]
        )
