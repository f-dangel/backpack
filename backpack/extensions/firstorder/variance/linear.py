from backpack.extensions.firstorder.gradient.linear import (
    GradLinear, GradLinearConcat
)
from backpack.extensions.firstorder.sum_grad_squared.linear import (
    SGSLinear, SGSLinearConcat
)
from .variance_base import VarianceBaseModule


class VarianceLinear(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradLinear(),
            sgs_extension=SGSLinear()
        )


class VarianceLinearConcat(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["weight"],
            grad_extension=GradLinearConcat(),
            sgs_extension=SGSLinearConcat()
        )
