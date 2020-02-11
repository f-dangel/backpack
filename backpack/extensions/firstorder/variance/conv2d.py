from backpack.extensions.firstorder.gradient.conv2d import GradConv2d
from backpack.extensions.firstorder.sum_grad_squared.conv2d import SGSConv2d

from .variance_base import VarianceBaseModule


class VarianceConv2d(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradConv2d(),
            sgs_extension=SGSConv2d(),
        )
