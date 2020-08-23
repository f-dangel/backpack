from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives

from .base import GradBaseModule


class GradConvTranspose1d(GradBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(), params=["bias", "weight"]
        )
