from backpack.core.layers import Conv2dConcat, LinearConcat
from backpack.extensions.backprop_extension import BackpropExtension
from torch.nn import BatchNorm1d, Conv2d, Linear

from . import batchnorm1d, conv2d, linear


class BatchGrad(BackpropExtension):
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
