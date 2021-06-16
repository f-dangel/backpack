"""batch_l2 extension for Conv."""
from torch import einsum

from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base
from backpack.utils import conv as convUtils


class BatchL2ConvND(BatchL2Base):
    """batch_l2 extension for Conv."""

    def __init__(self, N, params=None):
        """Initialization.

        Args:
            N: number of dimensions
            params: list of parameter names. Defaults to None.
        """
        super().__init__(params=params)
        self.N = N

    def bias(self, ext, module, g_inp, g_out, backproped):
        """batch_l2 for bias.

        Args:
            ext: extension
            module: module
            g_inp: input gradients
            g_out: output gradients
            backproped: backpropagation quantities

        Returns:
            batch_l2 for bias
        """
        C_axis = 1
        return convUtils.get_bias_gradient_factors(g_out[0], C_axis, self.N)

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
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, g_out[0], module, self.N
        )
        return einsum("nmi,nki,nmj,nkj->n", dE_dY, X, dE_dY, X)


class BatchL2Conv1d(BatchL2ConvND):
    """batch_l2 extension for Conv1d."""

    def __init__(self):
        """Initialization."""
        super().__init__(N=1, params=["bias", "weight"])


class BatchL2Conv2d(BatchL2ConvND):
    """batch_l2 extension for Conv2d."""

    def __init__(self):
        """Initialization."""
        super().__init__(N=2, params=["bias", "weight"])


class BatchL2Conv3d(BatchL2ConvND):
    """batch_l2 extension for Conv3d."""

    def __init__(self):
        """Initialization."""
        super().__init__(N=3, params=["bias", "weight"])
