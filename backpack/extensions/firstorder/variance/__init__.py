from backpack.extensions.backprop_extension import BackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat
from torch.nn import Linear, Conv2d

from . import linear, conv2d


class Variance(BackpropExtension):
    """
    Estimates the variance of the gradient using the samples in the minibatch.
    Is only meaningful is the individual functions are independent (no batchnorm).

    Stores the output in :code:`variance`, has the same dimension as the gradient.
    """
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

