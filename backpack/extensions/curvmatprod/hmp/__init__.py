"""Matrix-free multiplication with the block-diagonal Hessian."""

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
    """Matrix-free multiplication with the block-diagonal Hessian.

    Stores the multiplication function in :code:`hmp`.

    For a parameter of shape ``[...]`` the function receives and returns a tensor of
    shape ``[V, ...]``. Each vector slice across the leading dimension is multiplied
    with the block-diagonal Hessian.
    """

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
