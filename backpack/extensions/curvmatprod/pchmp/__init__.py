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

from . import activations, conv2d, dropout, flatten, linear, losses, padding, pooling


class PCHMP(BackpropExtension):
    def __init__(self, savefield="pchmp"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.PCHMPMSELoss(),
                CrossEntropyLoss: losses.PCHMPCrossEntropyLoss(),
                Linear: linear.PCHMPLinear(),
                MaxPool2d: pooling.PCHMPMaxpool2d(),
                AvgPool2d: pooling.PCHMPAvgPool2d(),
                ZeroPad2d: padding.PCHMPZeroPad2d(),
                Conv2d: conv2d.PCHMPConv2d(),
                Dropout: dropout.PCHMPDropout(),
                Flatten: flatten.PCHMPFlatten(),
                ReLU: activations.PCHMPReLU(),
                Sigmoid: activations.PCHMPSigmoid(),
                Tanh: activations.PCHMPTanh(),
            },
        )
