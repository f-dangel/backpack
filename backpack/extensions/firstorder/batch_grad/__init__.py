from torch.nn import (
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
)


class BatchGrad(BackpropExtension):
    """Individual gradients for each sample in a minibatch.

    Stores the output in ``grad_batch`` as a ``[N x ...]`` tensor,
    where ``N`` batch size and ``...`` is the shape of the gradient.

    Note: beware of scaling issue
        The `individual gradients` depend on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchGrad`` will return

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.

    The concept of individual gradients is only meaningful if the
    objective is a sum of independent functions (no batchnorm).

    """

    def __init__(self):
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
            },
        )
