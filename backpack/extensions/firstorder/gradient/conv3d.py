from backpack.core.derivatives.conv3d import Conv3DDerivatives

from .base import GradBaseModule


class GradConv3d(GradBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv3DDerivatives(), params=["bias", "weight"])
