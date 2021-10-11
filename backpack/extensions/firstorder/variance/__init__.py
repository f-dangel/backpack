"""Defines backpropagation extension for variance: Variance.

Defines module extension for each module.
"""
from torch.nn import (
    LSTM,
    RNN,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Embedding,
    Linear,
)

from backpack.extensions.firstorder.base import FirstOrderBackpropExtension

from . import (
    batchnorm_nd,
    conv1d,
    conv2d,
    conv3d,
    convtranspose1d,
    convtranspose2d,
    convtranspose3d,
    embedding,
    linear,
    rnn,
)


class Variance(FirstOrderBackpropExtension):
    """Estimates the variance of the gradient using the samples in the minibatch.

    Stores the output in ``variance``. Same dimension as the gradient.

    .. note::

        Beware of scaling issue

        The variance depends on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``Variance`` will return the variance of the vectors

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.
    """

    def __init__(self):
        """Initialization.

        Defines module extension for each module.
        """
        super().__init__(
            savefield="variance",
            module_exts={
                Linear: linear.VarianceLinear(),
                Conv1d: conv1d.VarianceConv1d(),
                Conv2d: conv2d.VarianceConv2d(),
                Conv3d: conv3d.VarianceConv3d(),
                ConvTranspose1d: convtranspose1d.VarianceConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.VarianceConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.VarianceConvTranspose3d(),
                RNN: rnn.VarianceRNN(),
                LSTM: rnn.VarianceLSTM(),
                BatchNorm1d: batchnorm_nd.VarianceBatchNormNd(),
                BatchNorm2d: batchnorm_nd.VarianceBatchNormNd(),
                BatchNorm3d: batchnorm_nd.VarianceBatchNormNd(),
                Embedding: embedding.VarianceEmbedding(),
            },
        )
