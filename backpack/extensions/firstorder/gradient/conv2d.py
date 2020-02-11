from backpack.core.derivatives.conv2d import Conv2DDerivatives

from .base import GradBaseModule


class GradConv2d(GradBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])
