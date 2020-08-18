"""
Matrix-free multiplication with the block-diagonal positive-curvature Hessian (PCH).
"""

from torch.nn import (
    AvgPool2d,
    Conv2d,
    CrossEntropyLoss,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import activations, conv2d, dropout, flatten, linear, losses, padding, pooling


class PCHMP(BackpropExtension):
    """
    Matrix-free multiplication with the block-diagonal positive-curvature Hessian (PCH).

    Stores the multiplication function in :code:`pchmp`.

    For a parameter of shape ``[...]`` the function receives and returns a tensor of
    shape ``[V, ...]``. Each vector slice across the leading dimension is multiplied
    with the block-diagonal positive curvature Hessian.

    The PCH is proposed in

    - `BDA-PCH: Block-Diagonal Approximation of Positive-Curvature Hessian for
      Training Neural Networks <https://arxiv.org/abs/1802.06502v2>`_
      by Sheng-Wei Chen, Chun-Nan Chou and Edward Y. Chang, 2018.

    There are different concavity-eliminating modifications which can be selected
    by the `modify` argument (`"abs"` or `"clip"`).

    .. note::

       The name positive-curvature Hessian may be misleading. While the PCH is
       always positive semi-definite (PSD), it does not refer to the projection of
       the exact Hessian on to the space of PSD matrices.
    """

    def __init__(self, savefield="pchmp", modify="clip"):
        super().__init__(
            savefield=savefield,
            fail_mode="ERROR",
            module_exts={
                MSELoss: losses.PCHMPMSELoss(),
                CrossEntropyLoss: losses.PCHMPCrossEntropyLoss(),
                Linear: linear.PCHMPLinear(),
                MaxPool2d: pooling.PCHMPMaxpool2d(),
                AvgPool2d: pooling.PCHMPAvgPool2d(),
                ZeroPad2d: padding.PCHMPZeroPad2d(),
                Conv2d: conv2d.PCHMPConv2d(),
                Dropout: dropout.PCHMPDropout(),
                Flatten: flatten.PCHMPFlatten(),
                ReLU: activations.PCHMPReLU(),
                Sigmoid: activations.PCHMPSigmoid(),
                Tanh: activations.PCHMPTanh(),
            },
        )
        self._modify = modify

    def get_modification_mode(self):
        return self._modify
