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


class HMP(BackpropExtension):
    def __init__(self, savefield="hmp"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.HMPMSELoss(),
                CrossEntropyLoss: losses.HMPCrossEntropyLoss(),
                Linear: linear.HMPLinear(),
                MaxPool2d: pooling.HMPMaxpool2d(),
                AvgPool2d: pooling.HMPAvgPool2d(),
                ZeroPad2d: padding.HMPZeroPad2d(),
                Conv2d: conv2d.HMPConv2d(),
                Dropout: dropout.HMPDropout(),
                Flatten: flatten.HMPFlatten(),
                ReLU: activations.HMPReLU(),
                Sigmoid: activations.HMPSigmoid(),
                Tanh: activations.HMPTanh(),
                BatchNorm1d: batchnorm1d.HMPBatchNorm1d(),
            },
        )
