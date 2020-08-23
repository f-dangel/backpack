from torch.nn import (
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
    conv1d,
    conv2d,
    conv3d,
    convtranspose1d,
    convtranspose2d,
    convtranspose3d,
    linear,
)


class Variance(BackpropExtension):
    """Estimates the variance of the gradient using the samples in the minibatch.

    Stores the output in ``variance``. Same dimension as the gradient.

    Note: beware of scaling issue
        The variance depends on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``Variance`` will return the variance of the vectors

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.
    """

    def __init__(self):
        super().__init__(
            savefield="variance",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.VarianceLinear(),
                Conv1d: conv1d.VarianceConv1d(),
                Conv2d: conv2d.VarianceConv2d(),
                Conv3d: conv3d.VarianceConv3d(),
                ConvTranspose1d: convtranspose1d.VarianceConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.VarianceConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.VarianceConvTranspose3d(),
            },
        )
