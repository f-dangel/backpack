from backpack.extensions.backprop_extension import BackpropExtension
from backpack.core.layers import Conv2dConcat, LinearConcat, Flatten
from torch.nn import Linear, Conv2d, Dropout, MaxPool2d, Tanh, Sigmoid, ReLU, CrossEntropyLoss, MSELoss, AvgPool2d, ZeroPad2d
from . import pooling, conv2d, linear, activations, losses, padding, dropout, flatten


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
                LinearConcat: linear.DiagHLinearConcat(),
                MaxPool2d: pooling.DiagHMaxPool2d(),
                AvgPool2d: pooling.DiagHAvgPool2d(),
                ZeroPad2d: padding.DiagHZeroPad2d(),
                Conv2d: conv2d.DiagHConv2d(),
                Conv2dConcat: conv2d.DiagHConv2dConcat(),
                Dropout: dropout.DiagHDropout(),
                Flatten: flatten.DiagHFlatten(),
                ReLU: activations.DiagHReLU(),
                Sigmoid: activations.DiagHSigmoid(),
                Tanh: activations.DiagHTanh(),
            }
        )
