from backpack.extensions.backprop_extension import BackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat
from torch.nn import Linear, Conv2d

from . import linear, conv2d


class BatchL2Grad(BackpropExtension):
    """
    The L2 norm of individual gradients in the minibatch.
    Is only meaningful is the individual functions are independent (no batchnorm).

    Stores the output in :code:`batch_l2`
    as a vector of the size as the minibatch.
    """
    def __init__(self):
        super().__init__(
            savefield="batch_l2",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.BatchL2Linear(),
                LinearConcat: linear.BatchL2LinearConcat(),
                Conv2d: conv2d.BatchL2Conv2d(),
                Conv2dConcat: conv2d.BatchL2Conv2dConcat(),
            }
        )

