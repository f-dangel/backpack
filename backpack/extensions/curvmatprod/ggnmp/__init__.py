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


class GGNMP(BackpropExtension):
    def __init__(self, savefield="ggnmp"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.GGNMPMSELoss(),
                CrossEntropyLoss: losses.GGNMPCrossEntropyLoss(),
                Linear: linear.GGNMPLinear(),
                MaxPool2d: pooling.GGNMPMaxpool2d(),
                AvgPool2d: pooling.GGNMPAvgPool2d(),
                ZeroPad2d: padding.GGNMPZeroPad2d(),
                Conv2d: conv2d.GGNMPConv2d(),
                Dropout: dropout.GGNMPDropout(),
                Flatten: flatten.GGNMPFlatten(),
                ReLU: activations.GGNMPReLU(),
                Sigmoid: activations.GGNMPSigmoid(),
                Tanh: activations.GGNMPTanh(),
                BatchNorm1d: batchnorm1d.GGNMPBatchNorm1d(),
            },
        )
