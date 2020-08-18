"""
Matrix-free multiplication with the block-diagonal generalized Gauss-Newton/Fisher.
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
    """
    Matrix-free Multiplication with the block-diagonal generalized Gauss-Newton/Fisher.

    Stores the multiplication function in :code:`ggnmp`.

    For a parameter of shape ``[...]`` the function receives and returns a tensor of
    shape ``[V, ...]``. Each vector slice across the leading dimension is multiplied
    with the block-diagonal GGN/Fisher.
    """

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
