from backpack.extensions.backprop_extension import BackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat
from torch.nn import Linear, Conv2d

from . import linear, conv2d


class SumGradSquared(BackpropExtension):
    """
    The sum of individual-gradients-squared, or second moment of the gradient.
    Is only meaningful is the individual functions are independent (no batchnorm).

    Stores the output in :code:`sum_grad_squared`, has the same dimension as the gradient.
    """

    def __init__(self):
        super().__init__(
            savefield="sum_grad_squared",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.SGSLinear(),
                LinearConcat: linear.SGSLinearConcat(),
                Conv2d: conv2d.SGSConv2d(),
                Conv2dConcat: conv2d.SGSConv2dConcat(),
            }
        )

