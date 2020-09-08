from torch import einsum

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.sum_grad_squared.sgs_base import SGSBase


class SGSLinear(SGSBase):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        """Compute second moments without expanding individual gradients.

        Overwrites the base class implementation that computes the gradient second
        moments from individual gradients. This approach is more memory-efficient.

        Note:
            For details, see page 12 (paragraph about "second moment") of the
            paper (https://arxiv.org/pdf/1912.10985.pdf).
        """
        return einsum("ni,nj->ij", (g_out[0] ** 2, module.input0 ** 2))
