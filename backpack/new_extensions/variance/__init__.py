from backpack.newbackpropextension import NewBackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat
from torch.nn import Linear, Conv2d

from . import linear, conv2d


class Variance(NewBackpropExtension):

    def __init__(self):
        super().__init__(
            savefield="variance",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.VarianceLinear(),
                LinearConcat: linear.VarianceLinearConcat(),
                Conv2d: conv2d.VarianceConv2d(),
                Conv2dConcat: conv2d.VarianceConv2dConcat(),
            }
        )

