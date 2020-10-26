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


class SumGradSquared(BackpropExtension):
    """The sum of individual-gradients-squared, or second moment of the gradient.

    Stores the output in ``sum_grad_squared``. Same dimension as the gradient.

    Note: beware of scaling issue
        The second moment depends on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``SumGradSquared`` will return the sum of the squared

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.
    """

    def __init__(self):
        super().__init__(
            savefield="sum_grad_squared",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.SGSLinear(),
                Conv1d: conv1d.SGSConv1d(),
                Conv2d: conv2d.SGSConv2d(),
                Conv3d: conv3d.SGSConv3d(),
                ConvTranspose1d: convtranspose1d.SGSConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.SGSConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.SGSConvTranspose3d(),
            },
        )
