from backpack.core.layers import Conv2dConcat, LinearConcat
from backpack.extensions.backprop_extension import BackpropExtension
from torch.nn import BatchNorm1d, Conv2d, Linear

from . import batchnorm1d, conv2d, linear


class BatchGrad(BackpropExtension):
    """
    The individual gradients for each sample in a minibatch.
    Is only meaningful is the individual functions are independent (no batchnorm).

    Stores the output in :code:`grad_batch` as a :code:`[N x ...]` tensor,
    where :code:`N` is the size of the minibatch and :code:`...`
    is the shape of the gradient.
    """
    def __init__(self):
        super().__init__(savefield="grad_batch",
                         fail_mode="WARNING",
                         module_exts={
                             Linear: linear.BatchGradLinear(),
                             LinearConcat: linear.BatchGradLinearConcat(),
                             Conv2d: conv2d.BatchGradConv2d(),
                             Conv2dConcat: conv2d.BatchGradConv2dConcat(),
                             BatchNorm1d: batchnorm1d.BatchGradBatchNorm1d(),
                         })
