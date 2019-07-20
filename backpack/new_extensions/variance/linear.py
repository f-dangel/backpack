from backpack.new_extensions.gradient.linear import GradLinear
from backpack.new_extensions.gradient.linear import GradLinearConcat
from backpack.new_extensions.sumgradsquared.linear import SGSLinear
from backpack.new_extensions.sumgradsquared.linear import SGSLinearConcat
from backpack.new_extensions.variance.base import VarianceBaseModule


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
