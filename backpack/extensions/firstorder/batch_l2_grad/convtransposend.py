"""batch_l2 extension for ConvTranspose."""
from torch import einsum

from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base
from backpack.utils import conv_transpose as convTransposeUtils


class BatchL2ConvTransposeND(BatchL2Base):
    """batch_l2 extension for ConvTranspose."""

    def weight(self, ext, module, g_inp, g_out, backproped):
        """batch_l2 for weight.

        Args:
            ext: extension
            module: module
            g_inp: input gradients
            g_out: output gradients
            backproped: backpropagation quantities

        Returns:
            batch_l2 for weight
        """
        X, dE_dY = convTransposeUtils.get_weight_gradient_factors(
            module.input0, g_out[0], module
        )
        return einsum("nmi,nki,nmj,nkj->n", dE_dY, X, dE_dY, X)


class BatchL2ConvTranspose1d(BatchL2ConvTransposeND):
    """batch_l2 extension for ConvTranspose1d."""

    def __init__(self):
        """Initialization."""
        super().__init__(["bias", "weight"], derivatives=ConvTranspose1DDerivatives())


class BatchL2ConvTranspose2d(BatchL2ConvTransposeND):
    """batch_l2 extension for ConvTranspose2d."""

    def __init__(self):
        """Initialization."""
        super().__init__(["bias", "weight"], derivatives=ConvTranspose2DDerivatives())


class BatchL2ConvTranspose3d(BatchL2ConvTransposeND):
    """batch_l2 extension for ConvTranspose3d."""

    def __init__(self):
        """Initialization."""
        super().__init__(["bias", "weight"], derivatives=ConvTranspose3DDerivatives())
