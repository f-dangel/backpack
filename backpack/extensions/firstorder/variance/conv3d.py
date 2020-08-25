from backpack.extensions.firstorder.gradient.conv3d import GradConv3d
from backpack.extensions.firstorder.sum_grad_squared.conv3d import SGSConv3d

from .variance_base import VarianceBaseModule


class VarianceConv3d(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradConv3d(),
            sgs_extension=SGSConv3d(),
        )
