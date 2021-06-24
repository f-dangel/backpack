"""
Matrix-free multiplication with the block-diagonal generalized Gauss-Newton/Fisher.
"""

from torch.nn import (
    ELU,
    SELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    BatchNorm1d,
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
from backpack.extensions.curvmatprod.ggnmp import (
    activations,
    batchnorm1d,
    convnd,
    convtransposend,
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
                MaxPool1d: pooling.GGNMPMaxPool1d(),
                MaxPool2d: pooling.GGNMPMaxPool2d(),
                MaxPool3d: pooling.GGNMPMaxPool3d(),
                AvgPool1d: pooling.GGNMPAvgPool1d(),
                AvgPool2d: pooling.GGNMPAvgPool2d(),
                AvgPool3d: pooling.GGNMPAvgPool3d(),
                ZeroPad2d: padding.GGNMPZeroPad2d(),
                Conv1d: convnd.GGNMPConv1d(),
                Conv2d: convnd.GGNMPConv2d(),
                Conv3d: convnd.GGNMPConv3d(),
                ConvTranspose1d: convtransposend.GGNMPConvTranspose1d(),
                ConvTranspose2d: convtransposend.GGNMPConvTranspose2d(),
                ConvTranspose3d: convtransposend.GGNMPConvTranspose3d(),
                Dropout: dropout.GGNMPDropout(),
                Flatten: flatten.GGNMPFlatten(),
                ReLU: activations.GGNMPReLU(),
                Sigmoid: activations.GGNMPSigmoid(),
                Tanh: activations.GGNMPTanh(),
                LeakyReLU: activations.GGNMPLeakyReLU(),
                LogSigmoid: activations.GGNMPLogSigmoid(),
                ELU: activations.GGNMPELU(),
                SELU: activations.GGNMPSELU(),
                BatchNorm1d: batchnorm1d.GGNMPBatchNorm1d(),
            },
        )
