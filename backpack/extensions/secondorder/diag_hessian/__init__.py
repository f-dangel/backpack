from torch.nn import (
    ELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    LeakyReLU,
    Linear,
    LogSigmoid,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    activations,
    conv1d,
    conv2d,
    conv3d,
    convtranspose1d,
    convtranspose2d,
    convtranspose3d,
    dropout,
    flatten,
    linear,
    losses,
    padding,
    pooling,
)


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
                MaxPool1d: pooling.DiagHMaxPool1d(),
                MaxPool2d: pooling.DiagHMaxPool2d(),
                AvgPool1d: pooling.DiagHAvgPool1d(),
                MaxPool3d: pooling.DiagHMaxPool3d(),
                AvgPool2d: pooling.DiagHAvgPool2d(),
                AvgPool3d: pooling.DiagHAvgPool3d(),
                ZeroPad2d: padding.DiagHZeroPad2d(),
                Conv1d: conv1d.DiagHConv1d(),
                Conv2d: conv2d.DiagHConv2d(),
                Conv3d: conv3d.DiagHConv3d(),
                ConvTranspose1d: convtranspose1d.DiagHConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.DiagHConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.DiagHConvTranspose3d(),
                Dropout: dropout.DiagHDropout(),
                Flatten: flatten.DiagHFlatten(),
                ReLU: activations.DiagHReLU(),
                Sigmoid: activations.DiagHSigmoid(),
                Tanh: activations.DiagHTanh(),
                LeakyReLU: activations.DiagHLeakyReLU(),
                LogSigmoid: activations.DiagHLogSigmoid(),
                ELU: activations.DiagHELU(),
            },
        )
