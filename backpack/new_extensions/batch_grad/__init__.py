from backpack.newbackpropextension import NewBackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat
from torch.nn import Linear, Conv2d

from . import linear, conv2d


class BatchGrad(NewBackpropExtension):

    def __init__(self):
        super().__init__(
            savefield="grad_batch",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.BatchGradLinear(),
                LinearConcat: linear.BatchGradLinearConcat(),
                Conv2d: conv2d.BatchGradConv2d(),
                Conv2dConcat: conv2d.BatchGradConv2dConcat(),
            }
        )
