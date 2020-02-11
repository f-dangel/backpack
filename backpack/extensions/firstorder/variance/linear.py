from backpack.extensions.firstorder.gradient.linear import GradLinear
from backpack.extensions.firstorder.sum_grad_squared.linear import SGSLinear

from .variance_base import VarianceBaseModule


class VarianceLinear(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradLinear(),
            sgs_extension=SGSLinear(),
        )
