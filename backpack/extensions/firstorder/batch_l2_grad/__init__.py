"""Contains BatchL2Grad.

Defines the backpropagation extension.
Within it, define the extension for each module.
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
from backpack.extensions.firstorder.batch_l2_grad import (
    batchnorm_nd,
    convnd,
    convtransposend,
    embedding,
    linear,
    rnn,
)


class BatchL2Grad(FirstOrderBackpropExtension):
    """The squared L2 norm of individual gradients in the minibatch.

    Stores the output in ``batch_l2`` as a tensor of size ``[N]``,
    where ``N`` is the batch size.

    .. note::

        Beware of scaling issue

        The individual L2 norm depends on the scaling of the overall function.
        Let ``fᵢ`` be the loss of the ``i`` th sample, with gradient ``gᵢ``.
        ``BatchL2Grad`` will return the L2 norm of

        - ``[g₁, …, gₙ]`` if the loss is a sum, ``∑ᵢ₌₁ⁿ fᵢ``,
        - ``[¹/ₙ g₁, …, ¹/ₙ gₙ]`` if the loss is a mean, ``¹/ₙ ∑ᵢ₌₁ⁿ fᵢ``.
    """

    def __init__(self):
        """Initialization.

        Define the extensions for each module.
        """
        super().__init__(
            savefield="batch_l2",
            module_exts={
                Linear: linear.BatchL2Linear(),
                Conv1d: convnd.BatchL2Conv1d(),
                Conv2d: convnd.BatchL2Conv2d(),
                Conv3d: convnd.BatchL2Conv3d(),
                ConvTranspose1d: convtransposend.BatchL2ConvTranspose1d(),
                ConvTranspose2d: convtransposend.BatchL2ConvTranspose2d(),
                ConvTranspose3d: convtransposend.BatchL2ConvTranspose3d(),
                RNN: rnn.BatchL2RNN(),
                LSTM: rnn.BatchL2LSTM(),
                BatchNorm1d: batchnorm_nd.BatchL2BatchNorm(),
                BatchNorm2d: batchnorm_nd.BatchL2BatchNorm(),
                BatchNorm3d: batchnorm_nd.BatchL2BatchNorm(),
                Embedding: embedding.BatchL2Embedding(),
            },
        )
