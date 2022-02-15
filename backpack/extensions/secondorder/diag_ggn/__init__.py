"""Module contains definitions of DiagGGN extensions.

Contains:
DiagGGN(BackpropExtension)
DiagGGNExact(DiagGGN)
DiagGGNMC(DiagGGN)
BatchDiagGGN(BackpropExtension)
BatchDiagGGNExact(BatchDiagGGN)
BatchDiagGGNMC(BatchDiagGGN)
"""
from torch import Tensor
from torch.nn import (
    ELU,
    LSTM,
    RNN,
    SELU,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveAvgPool3d,
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
from backpack.custom_module.permute import Permute
from backpack.custom_module.scale_module import ScaleModule
from backpack.custom_module.slicing import Slicing
from backpack.extensions.secondorder.base import SecondOrderBackpropExtension
from backpack.extensions.secondorder.hbp import LossHessianStrategy

from . import (
    activations,
    adaptive_avg_pool_nd,
    batchnorm_nd,
    conv1d,
    conv2d,
    conv3d,
    convtranspose1d,
    convtranspose2d,
    convtranspose3d,
    custom_module,
    dropout,
    embedding,
    flatten,
    linear,
    losses,
    pad,
    padding,
    permute,
    pooling,
    rnn,
    slicing,
)


class DiagGGN(SecondOrderBackpropExtension):
    """Base class for diagonal generalized Gauss-Newton/Fisher matrix."""

    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING,
    ]

    def __init__(self, loss_hessian_strategy: str, savefield: str):
        """Initialization.

        Args:
            loss_hessian_strategy: either LossHessianStrategy.EXACT or .SAMPLING
            savefield: the field where to save the calculated property

        Raises:
            ValueError: if chosen loss strategy is not valid.
        """
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy)
                + "Valid strategies: [{}]".format(self.VALID_LOSS_HESSIAN_STRATEGIES)
            )

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.DiagGGNCrossEntropyLoss(),
                Linear: linear.DiagGGNLinear(),
                MaxPool1d: pooling.DiagGGNMaxPool1d(),
                MaxPool2d: pooling.DiagGGNMaxPool2d(),
                AvgPool1d: pooling.DiagGGNAvgPool1d(),
                MaxPool3d: pooling.DiagGGNMaxPool3d(),
                AvgPool2d: pooling.DiagGGNAvgPool2d(),
                AvgPool3d: pooling.DiagGGNAvgPool3d(),
                ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv1d: conv1d.DiagGGNConv1d(),
                Conv2d: conv2d.DiagGGNConv2d(),
                Conv3d: conv3d.DiagGGNConv3d(),
                ConvTranspose1d: convtranspose1d.DiagGGNConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.DiagGGNConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.DiagGGNConvTranspose3d(),
                Dropout: dropout.DiagGGNDropout(),
                Flatten: flatten.DiagGGNFlatten(),
                ReLU: activations.DiagGGNReLU(),
                Sigmoid: activations.DiagGGNSigmoid(),
                Tanh: activations.DiagGGNTanh(),
                LeakyReLU: activations.DiagGGNLeakyReLU(),
                LogSigmoid: activations.DiagGGNLogSigmoid(),
                ELU: activations.DiagGGNELU(),
                SELU: activations.DiagGGNSELU(),
                Identity: custom_module.DiagGGNScaleModule(),
                ScaleModule: custom_module.DiagGGNScaleModule(),
                SumModule: custom_module.DiagGGNSumModule(),
                RNN: rnn.DiagGGNRNN(),
                LSTM: rnn.DiagGGNLSTM(),
                Permute: permute.DiagGGNPermute(),
                AdaptiveAvgPool1d: adaptive_avg_pool_nd.DiagGGNAdaptiveAvgPoolNd(1),
                AdaptiveAvgPool2d: adaptive_avg_pool_nd.DiagGGNAdaptiveAvgPoolNd(2),
                AdaptiveAvgPool3d: adaptive_avg_pool_nd.DiagGGNAdaptiveAvgPoolNd(3),
                BatchNorm1d: batchnorm_nd.DiagGGNBatchNormNd(),
                BatchNorm2d: batchnorm_nd.DiagGGNBatchNormNd(),
                BatchNorm3d: batchnorm_nd.DiagGGNBatchNormNd(),
                Embedding: embedding.DiagGGNEmbedding(),
                Pad: pad.DiagGGNPad(),
                Slicing: slicing.DiagGGNSlicing(),
            },
        )

    def accumulate_backpropagated_quantities(
        self, existing: Tensor, other: Tensor
    ) -> Tensor:  # noqa: D102
        return existing + other


class DiagGGNExact(DiagGGN):
    """Diagonal of the Generalized Gauss-Newton/Fisher.

    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`diag_ggn_exact`,
    has the same dimensions as the gradient.

    For a faster but less precise alternative,
    see :py:meth:`backpack.extensions.DiagGGNMC`.
    """

    def __init__(self):
        """Initialization. Chooses exact loss strategy and savefield diag_ggn_exact."""
        super().__init__(LossHessianStrategy.EXACT, "diag_ggn_exact")


class DiagGGNMC(DiagGGN):
    """Diagonal of the Generalized Gauss-Newton/Fisher.

    Uses a Monte-Carlo approximation of
    the Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`diag_ggn_mc`,
    has the same dimensions as the gradient.

    For a more precise but slower alternative,
    see :py:meth:`backpack.extensions.DiagGGNExact`.
    """

    def __init__(self, mc_samples: int = 1):
        """Initialization. Chooses sampling loss strategy and savefield diag_ggn_mc.

        Args:
            mc_samples: Number of Monte-Carlo samples. Default: ``1``.
        """
        self._mc_samples = mc_samples
        super().__init__(LossHessianStrategy.SAMPLING, "diag_ggn_mc")

    def get_num_mc_samples(self) -> int:
        """Returns number of Monte-Carlo samples.

        Returns:
            number of Monte-Carlo samples
        """
        return self._mc_samples


class BatchDiagGGN(SecondOrderBackpropExtension):
    """Base class for batched diagonal generalized Gauss-Newton/Fisher matrix."""

    VALID_LOSS_HESSIAN_STRATEGIES = [
        LossHessianStrategy.EXACT,
        LossHessianStrategy.SAMPLING,
    ]

    def __init__(self, loss_hessian_strategy: str, savefield: str):
        """Initialization.

        Args:
            loss_hessian_strategy: either LossHessianStrategy.EXACT or .SAMPLING
            savefield: name of variable where to save calculated quantity

        Raises:
            ValueError: if chosen loss strategy is not valid.
        """
        if loss_hessian_strategy not in self.VALID_LOSS_HESSIAN_STRATEGIES:
            raise ValueError(
                "Unknown hessian strategy: {}".format(loss_hessian_strategy)
                + "Valid strategies: [{}]".format(self.VALID_LOSS_HESSIAN_STRATEGIES)
            )

        self.loss_hessian_strategy = loss_hessian_strategy
        super().__init__(
            savefield=savefield,
            module_exts={
                MSELoss: losses.DiagGGNMSELoss(),
                CrossEntropyLoss: losses.DiagGGNCrossEntropyLoss(),
                Linear: linear.BatchDiagGGNLinear(),
                MaxPool1d: pooling.DiagGGNMaxPool1d(),
                MaxPool2d: pooling.DiagGGNMaxPool2d(),
                AvgPool1d: pooling.DiagGGNAvgPool1d(),
                MaxPool3d: pooling.DiagGGNMaxPool3d(),
                AvgPool2d: pooling.DiagGGNAvgPool2d(),
                AvgPool3d: pooling.DiagGGNAvgPool3d(),
                ZeroPad2d: padding.DiagGGNZeroPad2d(),
                Conv1d: conv1d.BatchDiagGGNConv1d(),
                Conv2d: conv2d.BatchDiagGGNConv2d(),
                Conv3d: conv3d.BatchDiagGGNConv3d(),
                ConvTranspose1d: convtranspose1d.BatchDiagGGNConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.BatchDiagGGNConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.BatchDiagGGNConvTranspose3d(),
                Dropout: dropout.DiagGGNDropout(),
                Flatten: flatten.DiagGGNFlatten(),
                ReLU: activations.DiagGGNReLU(),
                Sigmoid: activations.DiagGGNSigmoid(),
                Tanh: activations.DiagGGNTanh(),
                LeakyReLU: activations.DiagGGNLeakyReLU(),
                LogSigmoid: activations.DiagGGNLogSigmoid(),
                ELU: activations.DiagGGNELU(),
                SELU: activations.DiagGGNSELU(),
                Identity: custom_module.DiagGGNScaleModule(),
                ScaleModule: custom_module.DiagGGNScaleModule(),
                SumModule: custom_module.DiagGGNSumModule(),
                RNN: rnn.BatchDiagGGNRNN(),
                LSTM: rnn.BatchDiagGGNLSTM(),
                Permute: permute.DiagGGNPermute(),
                AdaptiveAvgPool1d: adaptive_avg_pool_nd.DiagGGNAdaptiveAvgPoolNd(1),
                AdaptiveAvgPool2d: adaptive_avg_pool_nd.DiagGGNAdaptiveAvgPoolNd(2),
                AdaptiveAvgPool3d: adaptive_avg_pool_nd.DiagGGNAdaptiveAvgPoolNd(3),
                BatchNorm1d: batchnorm_nd.BatchDiagGGNBatchNormNd(),
                BatchNorm2d: batchnorm_nd.BatchDiagGGNBatchNormNd(),
                BatchNorm3d: batchnorm_nd.BatchDiagGGNBatchNormNd(),
                Embedding: embedding.BatchDiagGGNEmbedding(),
                Pad: pad.DiagGGNPad(),
                Slicing: slicing.DiagGGNSlicing(),
            },
        )

    def accumulate_backpropagated_quantities(
        self, existing: Tensor, other: Tensor
    ) -> Tensor:  # noqa: D102
        return existing + other


class BatchDiagGGNExact(BatchDiagGGN):
    """Individual diagonal of the Generalized Gauss-Newton/Fisher.

    Uses the exact Hessian of the loss w.r.t. the model output.

    Stores the output in ``diag_ggn_exact_batch`` as a ``[N x ...]`` tensor,
    where ``N`` is the batch size and ``...`` is the shape of the gradient.
    """

    def __init__(self):
        """Initialization.

        Chooses exact loss strategy and savefield diag_ggn_exact_batch.
        """
        super().__init__(LossHessianStrategy.EXACT, "diag_ggn_exact_batch")


class BatchDiagGGNMC(BatchDiagGGN):
    """Individual diagonal of the Generalized Gauss-Newton/Fisher.

    Uses a Monte-Carlo approximation of
    the Hessian of the loss w.r.t. the model output.

    Stores the output in :code:`diag_ggn_mc_batch` as a ``[N x ...]`` tensor,
    where ``N`` is the batch size and ``...`` is the shape of the gradient.

    For a more precise but slower alternative,
    see :py:meth:`backpack.extensions.BatchDiagGGNExact`.
    """

    def __init__(self, mc_samples: int = 1):
        """Initialization.

        Chooses sampling loss strategy and savefield diag_ggn_mc_batch.

        Args:
            mc_samples: Number of Monte-Carlo samples. Default: ``1``.
        """
        self._mc_samples = mc_samples
        super().__init__(LossHessianStrategy.SAMPLING, "diag_ggn_mc_batch")

    def get_num_mc_samples(self) -> int:
        """Returns number of Monte-Carlo samples.

        Returns:
            number of Monte-Carlo samples
        """
        return self._mc_samples
