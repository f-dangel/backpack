"""
Curvature-matrix product backPACK extensions.

Those extension propagate additional information through the computation graph.
They are more expensive to run than a standard gradient backpropagation.

This extension does not compute information directly, but gives access to
functions to compute Matrix-Matrix products with Block-Diagonal approximations
of the curvature, such as the Block-diagonal Generalized Gauss-Newton
"""

from torch.nn import (
    AvgPool2d,
    BatchNorm1d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    activations,
    batchnorm1d,
    conv2d,
    dropout,
    flatten,
    linear,
    losses,
    padding,
    pooling,
)
from .hmp import HMP


class CMP(BackpropExtension):
    def __init__(self, curv_type, savefield="cmp"):
        self.curv_type = curv_type

        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.CMPMSELoss(),
                CrossEntropyLoss: losses.CMPCrossEntropyLoss(),
                Linear: linear.CMPLinear(),
                MaxPool2d: pooling.CMPMaxpool2d(),
                AvgPool2d: pooling.CMPAvgPool2d(),
                ZeroPad2d: padding.CMPZeroPad2d(),
                Conv2d: conv2d.CMPConv2d(),
                Dropout: dropout.CMPDropout(),
                Flatten: flatten.CMPFlatten(),
                ReLU: activations.CMPReLU(),
                Sigmoid: activations.CMPSigmoid(),
                Tanh: activations.CMPTanh(),
                BatchNorm1d: batchnorm1d.CMPBatchNorm1d(),
            },
        )

    def get_curv_type(self):
        return self.curv_type
