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
        has_additional_axes = g_out[0].dim() > 2

        if has_additional_axes:
            # TODO Compare `torch.einsum`, `opt_einsum.contract` and the base class
            # implementation: https://github.com/fKunstner/backpack-discuss/issues/111
            dE_dY = g_out[0].flatten(start_dim=1, end_dim=-2)
            X = module.input0.flatten(start_dim=1, end_dim=-2)
            return einsum("nmi,nmj,nki,nkj->ij", dE_dY, X, dE_dY, X)
        else:
            return einsum("ni,nj->ij", g_out[0] ** 2, module.input0**2)
