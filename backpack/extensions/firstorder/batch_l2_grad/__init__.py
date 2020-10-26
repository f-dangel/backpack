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


class BatchL2Grad(BackpropExtension):
    """The squared L2 norm of individual gradients in the minibatch.

    Stores the output in ``batch_l2`` as a tensor of size ``[N]``,
    where ``N`` is the batch size.

    Note: beware of scaling issue
        The individual L2 norm depends on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchL2Grad`` will return the L2 norm of

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.
    """

    def __init__(self):
        super().__init__(
            savefield="batch_l2",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.BatchL2Linear(),
                Conv1d: conv1d.BatchL2Conv1d(),
                Conv2d: conv2d.BatchL2Conv2d(),
                Conv3d: conv3d.BatchL2Conv3d(),
                ConvTranspose1d: convtranspose1d.BatchL2ConvTranspose1d(),
                ConvTranspose2d: convtranspose2d.BatchL2ConvTranspose2d(),
                ConvTranspose3d: convtranspose3d.BatchL2ConvTranspose3d(),
            },
        )
