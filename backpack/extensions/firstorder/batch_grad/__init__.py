"""Contains the backpropagation extension for grad_batch: BatchGrad.

It defines the module extension for each module.
"""
from typing import List, Union

from torch.nn import (
    RNN,
    BatchNorm1d,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Linear,
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    batchnorm1d,
    conv1d,
    conv2d,
    conv3d,
    conv_transpose1d,
    conv_transpose2d,
    conv_transpose3d,
    linear,
    rnn,
)


class BatchGrad(BackpropExtension):
    """Individual gradients for each sample in a minibatch.

    Stores the output in ``grad_batch`` as a ``[N x ...]`` tensor,
    where ``N`` batch size and ``...`` is the shape of the gradient.

    If ``subsampling`` is specified, ``N`` is replaced by the number of active
    samples.

    .. note::

        Beware of scaling issue

        The `individual gradients` depend on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchGrad`` will return

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

    The concept of individual gradients is only meaningful if the
    objective is a sum of independent functions (no batchnorm).
    """

    def __init__(self, subsampling: List[int] = None):
        """Initialization.

        Defines extension for each module.

        Args:
            subsampling: Indices of samples in the mini-batch for which individual
                gradients will be computed. Defaults to ``None`` (use all samples).
        """
        super().__init__(
            savefield="grad_batch",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.BatchGradLinear(),
                Conv1d: conv1d.BatchGradConv1d(),
                Conv2d: conv2d.BatchGradConv2d(),
                Conv3d: conv3d.BatchGradConv3d(),
                ConvTranspose1d: conv_transpose1d.BatchGradConvTranspose1d(),
                ConvTranspose2d: conv_transpose2d.BatchGradConvTranspose2d(),
                ConvTranspose3d: conv_transpose3d.BatchGradConvTranspose3d(),
                BatchNorm1d: batchnorm1d.BatchGradBatchNorm1d(),
                RNN: rnn.BatchGradRNN(),
            },
        )
        self._subsampling = subsampling

    def get_subsampling(self) -> Union[List[int], None]:
        """Get the indices of samples for which individual gradients are requested.

        Returns:
            List of indices containing the active samples in the mini-batch. ``None``
            means all samples will be considered.
        """
        return self._subsampling
