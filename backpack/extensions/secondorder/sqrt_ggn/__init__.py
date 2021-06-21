"""Defines base class and extensions for computing the GGN/Fisher matrix square root."""

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

from backpack.extensions.backprop_extension import BackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from backpack.extensions.secondorder.sqrt_ggn import (
    activations,
    convnd,
    convtransposend,
    dropout,
    flatten,
    linear,
    losses,
    padding,
    pooling,
)


class SqrtGGN(BackpropExtension):
    """Base class for extensions that compute the GGN/Fisher matrix square root."""

    def __init__(self, loss_hessian_strategy: str, savefield: str):
        """Store approximation for backpropagated object and where to save the result.

        Args:
            loss_hessian_strategy: Which approximation is used for the backpropagated
                loss Hessian. Must be ``'exact'`` or ``'sampling'``.
            savefield: Attribute under which the quantity is saved in a parameter.
        """
        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.SqrtGGNMSELoss(),
                CrossEntropyLoss: losses.SqrtGGNCrossEntropyLoss(),
                Linear: linear.SqrtGGNLinear(),
                MaxPool1d: pooling.SqrtGGNMaxPool1d(),
                MaxPool2d: pooling.SqrtGGNMaxPool2d(),
                AvgPool1d: pooling.SqrtGGNAvgPool1d(),
                MaxPool3d: pooling.SqrtGGNMaxPool3d(),
                AvgPool2d: pooling.SqrtGGNAvgPool2d(),
                AvgPool3d: pooling.SqrtGGNAvgPool3d(),
                ZeroPad2d: padding.SqrtGGNZeroPad2d(),
                Conv1d: convnd.SqrtGGNConv1d(),
                Conv2d: convnd.SqrtGGNConv2d(),
                Conv3d: convnd.SqrtGGNConv3d(),
                ConvTranspose1d: convtransposend.SqrtGGNConvTranspose1d(),
                ConvTranspose2d: convtransposend.SqrtGGNConvTranspose2d(),
                ConvTranspose3d: convtransposend.SqrtGGNConvTranspose3d(),
                Dropout: dropout.SqrtGGNDropout(),
                Flatten: flatten.SqrtGGNFlatten(),
                ReLU: activations.SqrtGGNReLU(),
                Sigmoid: activations.SqrtGGNSigmoid(),
                Tanh: activations.SqrtGGNTanh(),
                LeakyReLU: activations.SqrtGGNLeakyReLU(),
                LogSigmoid: activations.SqrtGGNLogSigmoid(),
                ELU: activations.SqrtGGNELU(),
                SELU: activations.SqrtGGNSELU(),
            },
        )


class SqrtGGNExact(SqrtGGN):
    """Exact matrix square root of the generalized Gauss-Newton/Fisher.

    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`sqrt_ggn_exact`, has shape ``[C, N, param.shape]``,
    where ``C`` is the model output dimension (number of classes for classification
    problems) and ``N`` is the batch size.

    For a faster but less precise alternative, see
    :py:meth:`backpack.extensions.SqrtGGNMC`.

    .. note::

        (Relation to the GGN/Fisher) For each parameter, ``param.sqrt_ggn_exact``
        can be viewed as a ``[C * N, param.numel()]`` matrix. Concatenating this
        matrix over all parameters results in a matrix ``Vᵀ``, which
        is the GGN/Fisher's matrix square root, i.e. ``G = V Vᵀ``.
    """

    def __init__(self):
        """Use exact loss Hessian and set savefield to ``sqrt_ggn_exact``."""
        super().__init__(LossHessianStrategy.EXACT, "sqrt_ggn_exact")
