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

    The function receives a tensor with trailing size identical to the
    parameter, and an additional leading dimension. Each slice across this leading
    dimension will be multiplied with the block-diagonal Hessian.

    Implements the procedures described by

    - `Modular Block-diagonal Curvature Approximations for Feedforward Architectures
      <https://arxiv.org/abs/1802.06502v2>`_
      by Felix Dangel, Stefan Harmeling, Philipp Hennig, 2020.
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
