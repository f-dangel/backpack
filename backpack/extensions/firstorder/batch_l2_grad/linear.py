"""Contains batch_l2 extension for Linear."""
from torch import einsum

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base


class BatchL2Linear(BatchL2Base):
    """batch_l2 extension for Linear."""

    def __init__(self):
        """Initialization."""
        super().__init__(params=["bias", "weight"], derivatives=LinearDerivatives())

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
        return einsum("ni,nj->n", g_out[0] ** 2, module.input0 ** 2)
