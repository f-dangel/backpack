"""Defines base class and extensions for computing the GGN/Fisher matrix square root."""

from typing import List, Union

from torch import Tensor
from torch.nn import (
    ELU,
    SELU,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    CrossEntropyLoss,
    Dropout,
    Embedding,
    Flatten,
    Identity,
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

from backpack.custom_module.branching import SumModule
from backpack.custom_module.pad import Pad
from backpack.custom_module.scale_module import ScaleModule
from backpack.custom_module.slicing import Slicing
from backpack.extensions.secondorder.base import SecondOrderBackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy
from backpack.extensions.secondorder.sqrt_ggn import (
    activations,
    batchnorm_nd,
    convnd,
    convtransposend,
    custom_module,
    dropout,
    embedding,
    flatten,
    linear,
    losses,
    pad,
    padding,
    pooling,
    slicing,
)


class SqrtGGN(SecondOrderBackpropExtension):
    """Base class for extensions that compute the GGN/Fisher matrix square root."""

    def __init__(
        self,
        loss_hessian_strategy: str,
        savefield: str,
        subsampling: Union[List[int], None],
    ):
        """Store approximation for backpropagated object and where to save the result.

        Args:
            loss_hessian_strategy: Which approximation is used for the backpropagated
                loss Hessian. Must be ``'exact'`` or ``'sampling'``.
            savefield: Attribute under which the quantity is saved in a parameter.
            subsampling: Indices of active samples. ``None`` uses the full mini-batch.
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
                Embedding: embedding.SqrtGGNEmbedding(),
                Identity: custom_module.SqrtGGNScaleModule(),
                ScaleModule: custom_module.SqrtGGNScaleModule(),
                SumModule: custom_module.SqrtGGNSumModule(),
                BatchNorm1d: batchnorm_nd.SqrtGGNBatchNormNd(),
                BatchNorm2d: batchnorm_nd.SqrtGGNBatchNormNd(),
                BatchNorm3d: batchnorm_nd.SqrtGGNBatchNormNd(),
                Pad: pad.SqrtGGNPad(),
                Slicing: slicing.SqrtGGNSlicing(),
            },
            subsampling=subsampling,
        )

    def get_loss_hessian_strategy(self) -> str:
        """Return the strategy used to represent the backpropagated loss Hessian.

        Returns:
            Loss Hessian strategy.
        """
        return self.loss_hessian_strategy

    def accumulate_backpropagated_quantities(
        self, existing: Tensor, other: Tensor
    ) -> Tensor:  # noqa: D102
        return existing + other


class SqrtGGNExact(SqrtGGN):
    """Exact matrix square root of the generalized Gauss-Newton/Fisher.

    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`sqrt_ggn_exact`, has shape ``[C, N, param.shape]``,
    where ``C`` is the model output dimension (number of classes for classification
    problems) and ``N`` is the batch size. If sub-sampling is enabled, ``N`` is
    replaced by the number of active samples, ``len(subsampling)``.

    For a faster but less precise alternative, see
    :py:meth:`backpack.extensions.SqrtGGNMC`.

    .. note::

        (Relation to the GGN/Fisher) For each parameter, ``param.sqrt_ggn_exact``
        can be viewed as a ``[C * N, param.numel()]`` matrix. Concatenating this
        matrix over all parameters results in a matrix ``Vᵀ``, which
        is the GGN/Fisher's matrix square root, i.e. ``G = V Vᵀ``.
    """

    def __init__(self, subsampling: List[int] = None):
        """Use exact loss Hessian, store results under ``sqrt_ggn_exact``.

        Args:
            subsampling: Indices of active samples. Defaults to ``None`` (use all
                samples in the mini-batch).
        """
        super().__init__(LossHessianStrategy.EXACT, "sqrt_ggn_exact", subsampling)


class SqrtGGNMC(SqrtGGN):
    """Approximate matrix square root of the generalized Gauss-Newton/Fisher.

    Uses a Monte-Carlo (MC) approximation of the Hessian of the loss w.r.t. the model
    output.

    Stores the output in :code:`sqrt_ggn_mc`, has shape ``[M, N, param.shape]``,
    where ``M`` is the number of Monte-Carlo samples and ``N`` is the batch size.
    If sub-sampling is enabled, ``N`` is replaced by the number of active samples,
    ``len(subsampling)``.

    For a more precise but slower alternative, see
    :py:meth:`backpack.extensions.SqrtGGNExact`.

    .. note::

        (Relation to the GGN/Fisher) For each parameter, ``param.sqrt_ggn_mc``
        can be viewed as a ``[M * N, param.numel()]`` matrix. Concatenating this
        matrix over all parameters results in a matrix ``Vᵀ``, which
        is the approximate GGN/Fisher's matrix square root, i.e. ``G ≈ V Vᵀ``.
    """

    def __init__(self, mc_samples: int = 1, subsampling: List[int] = None):
        """Approximate loss Hessian via MC and set savefield to ``sqrt_ggn_mc``.

        Args:
            mc_samples: Number of Monte-Carlo samples. Default: ``1``.
            subsampling: Indices of active samples. Defaults to ``None`` (use all
                samples in the mini-batch).
        """
        self._mc_samples = mc_samples
        super().__init__(LossHessianStrategy.SAMPLING, "sqrt_ggn_mc", subsampling)

    def get_num_mc_samples(self) -> int:
        """Return the number of MC samples used to approximate the loss Hessian.

        Returns:
            Number of Monte-Carlo samples.
        """
        return self._mc_samples
