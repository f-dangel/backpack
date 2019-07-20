from backpack.new_extensions.gradient.conv2d import GradConv2dConcat
from backpack.new_extensions.sumgradsquared.conv2d import SGSConv2dConcat
from backpack.new_extensions.gradient.conv2d import GradConv2d
from backpack.new_extensions.sumgradsquared.conv2d import SGSConv2d
from backpack.new_extensions.variance.base import VarianceBaseModule


class VarianceConv2d(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["bias", "weight"],
            grad_extension=GradConv2d(),
            sgs_extension=SGSConv2d()
        )


class VarianceConv2dConcat(VarianceBaseModule):
    def __init__(self):
        super().__init__(
            params=["weight"],
            grad_extension=GradConv2dConcat(),
            sgs_extension=SGSConv2dConcat()
        )
