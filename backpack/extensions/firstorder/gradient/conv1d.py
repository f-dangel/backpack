from backpack.core.derivatives.conv1d import Conv1DDerivatives

from .base import GradBaseModule


class GradConv1d(GradBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv1DDerivatives(), params=["bias", "weight"])
