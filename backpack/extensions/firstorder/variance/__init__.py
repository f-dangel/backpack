from torch.nn import Conv2d, Linear

from backpack.extensions.backprop_extension import BackpropExtension

from . import conv2d, linear


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
                Conv2d: conv2d.VarianceConv2d(),
            },
        )
