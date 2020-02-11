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


class DiagHessian(BackpropExtension):
    """
    Diagonal of the Hessian.

    Stores the output in :code:`diag_h`, has the same dimensions as the gradient.

    .. warning::

        Very expensive on networks with non-piecewise linear activations.

    """

    def __init__(self):
        super().__init__(
            savefield="diag_h",
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.DiagHMSELoss(),
                CrossEntropyLoss: losses.DiagHCrossEntropyLoss(),
                Linear: linear.DiagHLinear(),
                MaxPool2d: pooling.DiagHMaxPool2d(),
                AvgPool2d: pooling.DiagHAvgPool2d(),
                ZeroPad2d: padding.DiagHZeroPad2d(),
                Conv2d: conv2d.DiagHConv2d(),
                Dropout: dropout.DiagHDropout(),
                Flatten: flatten.DiagHFlatten(),
                ReLU: activations.DiagHReLU(),
                Sigmoid: activations.DiagHSigmoid(),
                Tanh: activations.DiagHTanh(),
            },
        )
