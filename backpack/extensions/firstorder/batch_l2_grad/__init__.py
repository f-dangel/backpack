from torch.nn import Conv2d, Linear

from backpack.extensions.backprop_extension import BackpropExtension

from . import conv2d, linear


class BatchL2Grad(BackpropExtension):
    """
    The squared L2 norm of individual gradients in the minibatch.
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
                Conv2d: conv2d.BatchL2Conv2d(),
            },
        )
