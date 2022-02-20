"""Define BackPACK extensions based on the Hessian diagonal.

- Hessian diagonal
- Per-sample (individual) Hessian diagonal
"""
from torch.nn import (
    ELU,
    SELU,
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

from backpack.custom_module.pad import Pad
from backpack.custom_module.slicing import Slicing
from backpack.extensions.secondorder.base import SecondOrderBackpropExtension

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
    pad,
    padding,
    pooling,
    slicing,
)


class DiagHessian(SecondOrderBackpropExtension):
    """BackPACK extension that computes the Hessian diagonal.

    Stores the output in :code:`diag_h`, has the same dimensions as the gradient.

    .. warning::

        Very expensive on networks with non-piecewise linear activations.

    """

    def __init__(self):
        """Store savefield and mappings between layers and module extensions."""
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
                SELU: activations.DiagHSELU(),
                Pad: pad.DiagHPad(),
                Slicing: slicing.DiagHSlicing(),
            },
        )


class BatchDiagHessian(SecondOrderBackpropExtension):
    """BackPACK extensions that computes the per-sample (individual) Hessian diagonal.

    Stores the output in ``diag_h_batch`` as a ``[N x ...]`` tensor,
    where ``N`` is the batch size and ``...`` is the parameter shape.

    .. warning::

        Very expensive on networks with non-piecewise linear activations.

    """

    def __init__(self):
        """Store savefield and mappings between layers and module extensions."""
        super().__init__(
            savefield="diag_h_batch",
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.DiagHMSELoss(),
                CrossEntropyLoss: losses.DiagHCrossEntropyLoss(),
                Linear: linear.BatchDiagHLinear(),
                MaxPool1d: pooling.DiagHMaxPool1d(),
                MaxPool2d: pooling.DiagHMaxPool2d(),
                AvgPool1d: pooling.DiagHAvgPool1d(),
                MaxPool3d: pooling.DiagHMaxPool3d(),
                AvgPool2d: pooling.DiagHAvgPool2d(),
                AvgPool3d: pooling.DiagHAvgPool3d(),
                ZeroPad2d: padding.DiagHZeroPad2d(),
                Conv1d: conv1d.BatchDiagHConv1d(),
                Conv2d: conv2d.BatchDiagHConv2d(),
                Conv3d: conv3d.BatchDiagHConv3d(),
                ConvTranspose1d: convtranspose1d.BatchDiagHConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.BatchDiagHConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.BatchDiagHConvTranspose3d(),
                Dropout: dropout.DiagHDropout(),
                Flatten: flatten.DiagHFlatten(),
                ReLU: activations.DiagHReLU(),
                Sigmoid: activations.DiagHSigmoid(),
                Tanh: activations.DiagHTanh(),
                LeakyReLU: activations.DiagHLeakyReLU(),
                LogSigmoid: activations.DiagHLogSigmoid(),
                ELU: activations.DiagHELU(),
                SELU: activations.DiagHSELU(),
                Pad: pad.DiagHPad(),
                Slicing: slicing.DiagHSlicing(),
            },
        )
